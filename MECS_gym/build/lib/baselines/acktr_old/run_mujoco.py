#!/usr/bin/env python3

import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.acktr_old.acktr_cont import learn
from baselines.acktr_old.policies import GaussianMlpPolicy
from baselines.acktr_old.value_functions import NeuralNetValueFunction
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def train(env_id, num_timesteps, seed):

    env = make_mujoco_env(env_id, seed)
    eval_env = make_mujoco_env(env_id, seed)

    sess = tf.InteractiveSession()
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    with tf.variable_scope("vf"):
        vf = NeuralNetValueFunction(ob_dim, ac_dim)
    with tf.variable_scope("pi"):
        policy = GaussianMlpPolicy(ob_dim, ac_dim)

    # log_dir = './result/%s'%(args.alg)

    learn(env, policy=policy, vf=vf,
          gamma=0.99, lam=0.97, timesteps_per_batch=2500,
          desired_kl=0.002,
          num_timesteps=num_timesteps, animate=False, eval_env=eval_env)

    env.close()

def main():
    Mujoco_Envs = ['HalfCheetah-v1', 'Hopper-v1',
                   'Walker2d-v1', 'Ant-v1', 'Humanoid-v1', 'Humanoid(rllab)', 'BipedalWalker-v2',
                   'BipedalWalkerHardcore-v2', 'Swimmer-v1', 'Pendulum-v0', 'Reacher-v1', 'HumanoidStandup-v1',
                   'InvertedDoublePendulum-v1', 'InvertedPendulum-v1', ]

    parser = arg_parser()
    parser.add_argument('--envn', help='environment ID', type=int, default=3)
    parser.add_argument('--evalenv', help='use Eval Env', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=float, default=3e6),
    args, _ = parser.parse_known_args()
    env_name = Mujoco_Envs[args.envn]
    # log_dir = '/home/han/Downloads/log_ppo/Mujoco/%s_leng%d_clip%.1f'%(args.alg, args.leng,args.eps)
    # log_dir = '/home/han/Downloads/log_ppo/Mujoco_test/leng%d_clip%.1f' % (args.leng, args.eps)
    log_dir = '/home/wisrl/Downloads/log_ppo/Mujoco/acktr'

    # log_dir = './result/acktr'

    log_dir += '/%s' % env_name


    if env_name == 'Humanoid-v1' or env_name == 'Humanoid(rllab)' or env_name == 'HumanoidStandup-v1':
        args.num_timesteps = 1e7
    if env_name == 'Ant-v1':
        args.num_timesteps = 5e6
    if env_name == 'Hopper-v1':
        args.num_timesteps = 1e6
    print("args")
    print(args)
    dir = log_dir + '/iter%d' % args.seed
    logger.configure(dir=dir)
    train(env_name, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == "__main__":
    main()
