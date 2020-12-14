import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            n_initial_exploration_steps=10000,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = int(n_epochs)
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._policy = None
        self._pool = None

    def _train(self, env, policy, initial_exploration_policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training(env, policy, pool)
        if initial_exploration_policy is None:
            self.sampler.initialize(env, policy, pool)
            initial_exploration_done = True
        else:
            self.sampler.initialize(env, initial_exploration_policy, pool)
            initial_exploration_done = False

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    if not initial_exploration_done:
                        if self._epoch_length * epoch >= self._n_initial_exploration_steps:
                            self.sampler.set_policy(policy)
                            initial_exploration_done = True
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            self.sampler.terminate()

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self.sampler._max_path_length, self._eval_n_episodes,
                            )

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]
        eval_obs = np.vstack([path['observations'] for path in paths])
        cloud_state = np.vstack([path['env_infos']['cloud_cpu_used'] for path in paths])
        print(eval_obs.shape)
        print(cloud_state)

        ### before (for ~v5)
        # eval_edge_s = np.transpose(eval_obs)[:40].reshape(-1, 8, len(eval_obs))
        # eval_cloud_s = np.transpose(eval_obs)[40:]
        # eval_edge_queue = eval_edge_s[2]-eval_edge_s[1]  # shape (8, episode length)
        # eval_edge_cpu = eval_edge_s[3]
        # eval_workload = eval_edge_s[4]
        # eval_cloud_queue = eval_cloud_s[2]
        # eval_cloud_cpu = eval_cloud_s[3]
        # eval_edge_queue_avg = eval_edge_queue[:3].mean()  # shape (,)
        # eval_cloud_queue_avg = eval_cloud_queue.mean()  # float
        #
        # eval_edge_power = 10 * (40*eval_edge_cpu.sum(axis=0) * (10 ** 9) / 10) ** 3  # shape (5000,)
        # eval_cloud_power = 54 * (216*eval_cloud_cpu * (10 ** 9) / 54) ** 3  # shape (5000,)
        #
        # eval_edge_power_avg = eval_edge_power.mean()
        # eval_cloud_power_avg = eval_cloud_power.mean()
        #
        # eval_power = eval_edge_power_avg + eval_cloud_power_avg
        eval_edge_s = np.transpose(eval_obs)[:15].reshape(-1, 3, len(eval_obs))
        # eval_edge_s = np.transpose(eval_obs)[:40].reshape(-1, 8, len(eval_obs))
        eval_edge_queue = eval_edge_s[2]-eval_edge_s[1]  # shape (8, episode length)
        eval_edge_cpu = eval_edge_s[3]
        eval_workload = eval_edge_s[4]


        eval_cloud_cpu = cloud_state[0]
        eval_edge_queue_avg = eval_edge_queue[:3].mean()  # shape (,)
        # eval_edge_queue_avg = eval_edge_queue.mean()  # shape (,)

        eval_edge_power = 10 * (40*eval_edge_cpu.sum(axis=0) * (10 ** 9) / 10) ** 3  # shape (5000,)
        eval_cloud_power = 54 * (216*eval_cloud_cpu * (10 ** 9) / 54) ** 3  # shape (5000,)

        eval_edge_power_avg = eval_edge_power.mean()
        eval_cloud_power_avg = eval_cloud_power.mean()

        eval_power = eval_edge_power_avg + eval_cloud_power_avg

        for i in range(int(len(eval_edge_power)/100)):
            start_ = i * 100
            end_ = (i+1) * 100
            logger.record_tabular("eval_q_1_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[0, start_:end_]))
            logger.record_tabular("eval_q_2_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[1, start_:end_]))
            logger.record_tabular("eval_q_3_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[2, start_:end_]))
            # logger.record_tabular("eval_q_4_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[3, start_:end_]))
            # logger.record_tabular("eval_q_5_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[4, start_:end_]))
            # logger.record_tabular("eval_q_6_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[5, start_:end_]))
            # logger.record_tabular("eval_q_7_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[6, start_:end_]))
            # logger.record_tabular("eval_q_8_%02d%02d"%(i,i+1), np.mean(eval_edge_queue[7, start_:end_]))
            # logger.record_tabular("eval_q_var_%02d%02d"%(i,i+1), np.mean(np.var(eval_edge_queue[:8, start_:end_], axis=0)))
            logger.record_tabular("eval_q_var_%02d%02d"%(i,i+1), np.mean(np.var(eval_edge_queue[:3, start_:end_], axis=0)))
        logger.record_tabular("eval_q_1_all", np.mean(eval_edge_queue[0,:]))
        logger.record_tabular("eval_q_2_all", np.mean(eval_edge_queue[1,:]))
        logger.record_tabular("eval_q_3_all", np.mean(eval_edge_queue[2,:]))
        # logger.record_tabular("eval_q_4_all", np.mean(eval_edge_queue[3,:]))
        # logger.record_tabular("eval_q_5_all", np.mean(eval_edge_queue[4,:]))
        # logger.record_tabular("eval_q_6_all", np.mean(eval_edge_queue[5,:]))
        # logger.record_tabular("eval_q_7_all", np.mean(eval_edge_queue[6,:]))
        # logger.record_tabular("eval_q_8_all", np.mean(eval_edge_queue[7,:]))
        # logger.record_tabular("eval_q_var_all", np.mean(np.var(eval_edge_queue[:8,:],axis=0)))
        logger.record_tabular("eval_q_var_all", np.mean(np.var(eval_edge_queue[:3,:],axis=0)))
        logger.record_tabular("eval_edge_queue_avg", eval_edge_queue_avg)
        logger.record_tabular("eval_power", eval_power)
        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        iteration = epoch*self._epoch_length
        batch = self.sampler.random_batch()
        self.log_diagnostics(iteration, batch)

    @abc.abstractmethod
    def log_diagnostics(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self._env = env
        if self._eval_n_episodes > 0:
            # TODO: This is horrible. Don't do this. Get rid of this.
            import tensorflow as tf
            with tf.variable_scope("low_level_policy", reuse=True):
                self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return self._pool
