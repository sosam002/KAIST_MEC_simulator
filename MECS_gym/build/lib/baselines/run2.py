import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    Mujoco_Envs = ['HalfCheetah-v1', 'Hopper-v1',
                   'Walker2d-v1', 'Ant-v1', 'Humanoid-v1', 'Humanoid(rllab)','BipedalWalker-v2',
                   'BipedalWalkerHardcore-v2', 'Swimmer-v1', 'Pendulum-v0', 'Reacher-v1', 'HumanoidStandup-v1', 'InvertedDoublePendulum-v1', 'InvertedPendulum-v1',]

    parser = arg_parser()
    parser.add_argument('--envn', help='environment ID', type=int, default=11)
    parser.add_argument('--leng', help='ER length', type=int, default=64)
    parser.add_argument('--eps', help='Clipping factor', type=float, default=0.4)
    parser.add_argument('--epscut', help='Clipping factor', type=float, default=0.1)
    parser.add_argument('--trrho', help='Clipping factor', type=float, default=1.0)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=3e-4)
    parser.add_argument('--constlr', help='Clipping factor', type=int, default=0)
    parser.add_argument('--rgae', help='Clipping factor', type=int, default=1)
    parser.add_argument('--adaptive_kl', help='Clipping factor', type=int, default=1)
    parser.add_argument('--dtarg', help='Clipping factor', type=float, default=0.001)
    parser.add_argument('--num_repeat', help='Repeat Number', type=int, default=1)
    parser.add_argument('--useadv', help='Clipping factor', type=int, default=0)
    parser.add_argument('--vtrace', help='use V-trace adv', type=int, default=1)
    parser.add_argument('--gradclip', help='Clipping factor', type=int, default=0)
    parser.add_argument('--numenv', help='Clipping factor', type=int, default=1)
    parser.add_argument('--evalenv', help='use Eval Env', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2_AMBER5_clipdim2')
    parser.add_argument('--num_timesteps', type=float, default=5e7),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    args, _ = parser.parse_known_args()
    parser.add_argument('--env', help='environment ID', type=str, default=Mujoco_Envs[args.envn])
    # log_dir = '/home/han/Downloads/log_ppo/Mujoco/%s_leng%d_clip%.1f'%(args.alg, args.leng,args.eps)
    # log_dir = '/home/han/Downloads/log_ppo/Mujoco_test/leng%d_clip%.1f' % (args.leng, args.eps)
    log_dir = '/home/wisrl/Downloads/log_ppo/Mujoco/%s_leng%d_clip%s'%(args.alg, args.leng, str(args.eps))

    log_dir = './test/%s_leng%d_clip%s'%(args.alg, args.leng, str(args.eps))
    if args.vtrace==1:
        log_dir+='_vtr'
    elif args.vtrace==4:
        log_dir+='_vtr4_tr%.1f'%args.trrho
    if args.useadv:
        log_dir+='_useadv'
    if args.adaptive_kl:
        log_dir+='_adap_kl'
        log_dir +='_dtarg%s'%(str(args.dtarg))
    if args.rgae:
        log_dir+='_rgae'
    if args.gradclip:
        log_dir += '_gradclip0.5'
    if args.constlr:
        log_dir += '_constlr'
    if 'AMBER' in args.alg:
        log_dir += '_clipcut%s'%(str(args.epscut))
    log_dir += '_minlr0.0001'

    log_dir += '/%s'%Mujoco_Envs[args.envn]
    parser.add_argument('--log_dir', help='Algorithm', type=str, default=log_dir)
    return parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def train(args, extra_args, seed):
    env_type, env_id = get_env_type(args.env)
    env_type='mujoco'
    print('env_type: {}'.format(env_type))
    total_timesteps = int(args.num_timesteps)
    # seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs["ERlen"] = args.leng
    alg_kwargs["vtrace"] = args.vtrace
    alg_kwargs["useadv"] = args.useadv
    if args.constlr:
        alg_kwargs["lr"] = args.lr
    alg_kwargs["dtarg"] = args.dtarg
    alg_kwargs["adaptive_kl"] = args.adaptive_kl
    alg_kwargs["trunc_rho"] = args.trrho
    alg_kwargs["rgae"] = args.rgae
    alg_kwargs["clipcut"] = args.epscut
    alg_kwargs["cliprange"] = args.eps
    if args.gradclip:
        alg_kwargs["max_grad_norm"] = 0.5
    alg_kwargs.update(extra_args)

    env, sess, eval_env = build_env(args, seed)


    if not args.evalenv:
        eval_env = None

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env, sess, eval_env


def build_env(args, seed):
    nenv = 1
    alg = args.alg
    # seed = args.seed
    seed = int(np.random.rand(1)*101000)
    print(seed)

    env_type, env_id = get_env_type(args.env)
    set_global_seeds(seed)
    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        # config = tf.ConfigProto(allow_soft_placement=True,
        #                        intra_op_parallelism_threads=1,
        #                        inter_op_parallelism_threads=1)
        # config.gpu_options.allow_growth = True
        # get_session(config=config)
        sess = tf.InteractiveSession()
        # env = VecNormalize(make_vec_env(env_id, env_type, 1, seed, reward_scale=args.reward_scale))

        env = make_vec_env(env_id, env_type, args.numenv, seed, reward_scale=args.reward_scale)
        evalenv = make_vec_env(env_id, env_type, args.numenv, seed, reward_scale=args.reward_scale)

        # if env_type == 'mujoco':
        #     env = VecNormalize(env)
        #     evalenv = VecNormalizeEval(evalenv)
        #     evalenv.ob_rms = env.ob_rms
        #     evalenv.ret_rms = env.ret_rms

    return env, sess, evalenv


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = 'mujoco'
        # for g, e in _game_envs.items():
        #     if env_id in e:
        #         env_type = g
        #         break
        # assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()

    # if args.env == 'Humanoid-v1' or args.env == 'Humanoid(rllab)' or args.env == 'HumanoidStandup-v1':
    #     args.num_timesteps = 1e7
    # if args.env == 'Ant-v1':
    #     args.num_timesteps = 5e6
    extra_args = parse_cmdline_kwargs(unknown_args)
    print("args")
    print(args)
    if args.num_repeat==1:
        dir = args.log_dir + '/iter%d' % args.seed
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            logger.configure(dir=dir)
        else:
            logger.configure(dir=dir, format_strs=[])
            rank = MPI.COMM_WORLD.Get_rank()

        model, env, sess, evalenv = train(args, extra_args, args.seed)
        env.close()
        evalenv.close()
        tf.reset_default_graph()
        sess.close()
    else:
        for seed in range(args.num_repeat):
            dir = args.log_dir + '/iter%d'%seed
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                rank = 0
                logger.configure(dir=dir)
            else:
                logger.configure(dir=dir, format_strs=[])
                rank = MPI.COMM_WORLD.Get_rank()

            model, env, sess, evalenv = train(args, extra_args, seed)
            env.close()
            evalenv.close()
            tf.reset_default_graph()
            sess.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

if __name__ == '__main__':
    main()
