import tensorflow as tf

from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger
from importlib import import_module
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from baselines.bench import Monitor
from baselines import constants
from gym.envs.registration import register

def make_vec_env(env_id, seed):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    env = gym.make(env_id)
    env.seed(seed)
    def make_thunk(env):
        return lambda: env
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), '0'), allow_early_resets=True)
    set_global_seeds(seed)
    return DummyVecEnv([make_thunk(env)])


import argparse
def disc_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    Mujoco_Envs = ['MECS-v1','MECS-v2','MECS-v3']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envn', help='environment ID', type=int, default=2)
    parser.add_argument('--leng', help='Replay length', type=int, default=64)
    parser.add_argument('--clip', help='Clipping factor', type=float, default=0.4)
    parser.add_argument('--batchlim', help='Batch limiting factor', type=float, default=0.1)
    parser.add_argument('--jtarg', help='Target IS constant', type=float, default=0.001)
    parser.add_argument('--fixstd', help='use Eval Env', type=int, default=1)
    parser.add_argument('--logstd_init', help='use Eval Env', type=int, default=0)
    parser.add_argument('--cost_type', help='use Eval Env', type=int, default=500000)
    parser.add_argument('--gaev', help='use GAE-V', type=int, default=1)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='DISC')
    parser.add_argument('--num_timesteps', type=float, default=5e7),
    args, _ = parser.parse_known_args()
    parser.add_argument('--env', help='environment ID', type=str, default=Mujoco_Envs[args.envn])

    log_dir = '/home/wisrl/Downloads/log_ppo/%s' % (args.alg)

    log_dir += '/%s'%Mujoco_Envs[args.envn]
    log_dir += '_c%s'%args.cost_type
    log_dir += '_f%s'%str(args.fixstd)
    log_dir += '_s%s'%str(args.logstd_init)
    parser.add_argument('--log_dir', help='log dir', type=str, default=log_dir)
    return parser

def train(args, seed):
    env_type='mujoco'
    env_id = args.env
    print('env_type: {}'.format(env_type))
    total_timesteps = int(args.num_timesteps)

    learn = import_module('.'.join(['baselines', args.alg, args.alg])).learn

    alg_kwargs={}
    alg_kwargs["lr"] = lambda f: 3e-4 * f
    alg_kwargs["value_network"] = 'copy'
    alg_kwargs["replay_length"] = args.leng
    alg_kwargs["vtrace"] = args.gaev
    alg_kwargs["J_targ"] = args.jtarg
    alg_kwargs["batchlim"] = args.batchlim
    alg_kwargs["cliprange"] = args.clip
    alg_kwargs['network'] = 'mlp'
    set_global_seeds(seed)

    sess = tf.InteractiveSession()

    env = make_vec_env(env_id, seed)
    eval_env = make_vec_env(env_id, seed)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        total_timesteps=total_timesteps,
        fixstd=args.fixstd,
        logstd_init=args.logstd_init,
        sess=sess,
        **alg_kwargs
    )
    return model, env, sess, eval_env


def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = disc_arg_parser()
    args, _ = arg_parser.parse_known_args()

    constants.COST_TYPE = args.cost_type

    register(
        id='MECS-v1',
        entry_point='baselines.environment_V_sweep:MEC_v1',
        max_episode_steps=5000,
    )

    register(
        id='MECS-v2',
        entry_point='baselines.env_V_sweep_v2:MEC_v2',
        max_episode_steps=5000,
    )

    register(
        id='MECS-v3',
        entry_point='baselines.env_V_sweep_v3:MEC_v3',
        max_episode_steps=5000,
    )

    dir = args.log_dir + '/iter%d' % args.seed
    logger.configure(dir=dir)

    model, env, sess, evalenv = train(args, args.seed)
    env.close()
    evalenv.close()

if __name__ == '__main__':
    main()
