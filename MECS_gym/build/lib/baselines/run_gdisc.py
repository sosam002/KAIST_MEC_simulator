import tensorflow as tf

from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger
from importlib import import_module
import gym
import os
from baselines.bench import Monitor

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


def disc_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Ant-v1')
    parser.add_argument('--leng', help='Replay length', type=int, default=64)
    parser.add_argument('--clip', help='Clipping factor', type=float, default=0.4)
    parser.add_argument('--batchlim', help='Batch limiting factor', type=float, default=0.1)
    parser.add_argument('--jtarg', help='Target IS constant', type=float, default=0.001)
    parser.add_argument('--gaev', help='use GAE-V', type=int, default=1)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='GDISC')
    parser.add_argument('--num_timesteps', type=float, default=3e6),
    args, _ = parser.parse_known_args()

    log_dir = './test/%s_leng%d_clip%s_batchlim%s/%s'%(args.alg, args.leng, str(args.clip),str(args.batchlim),args.env)
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
        **alg_kwargs
    )
    return model, env, sess, eval_env


def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = disc_arg_parser()
    args, _ = arg_parser.parse_known_args()

    if args.env == 'Humanoid-v1' or args.env == 'HumanoidStandup-v1':
        args.num_timesteps = 1e7

    dir = args.log_dir + '/iter%d' % args.seed
    logger.configure(dir=dir)

    model, env, sess, evalenv = train(args, args.seed)
    env.close()
    evalenv.close()

if __name__ == '__main__':
    main()
