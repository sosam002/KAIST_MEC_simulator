import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
# from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos import SAC
from sac.envs import (
    GymEnv,
    GymEnvDelayed,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv,
)

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants_re_1 import parse_domain_and_task, get_variants, get_variants_delayed
from sac.envs import constants
from gym.envs.registration import register


DELAY_CONST = 20
ENVIRONMENTS = {
    'swimmer-gym': {
        'default': lambda: GymEnv('Swimmer-v1'),
        'delayed': lambda: GymEnvDelayed('Swimmer-v1', delay = DELAY_CONST),
    },
    'swimmer-rllab': {
        'default': SwimmerEnv,
        'multi-direction': MultiDirectionSwimmerEnv,
    },
    'ant': {
        'default': lambda: GymEnv('Ant-v1'),
        'multi-direction': MultiDirectionAntEnv,
        'cross-maze': CrossMazeAntEnv,
        'delayed': lambda: GymEnvDelayed('Ant-v1', delay = DELAY_CONST),
    },
    'humanoid-gym': {
        'default': lambda: GymEnv('Humanoid-v1'),
        'delayed': lambda: GymEnvDelayed('Humanoid-v1', delay = DELAY_CONST),
    },
    'humanoid-rllab': {
        'default': HumanoidEnv,
        'multi-direction': MultiDirectionHumanoidEnv,
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1'),
        'delayed': lambda: GymEnvDelayed('Hopper-v1', delay = DELAY_CONST),
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1'),
        'delayed': lambda: GymEnvDelayed('HalfCheetah-v1', delay = DELAY_CONST),
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1'),
        'delayed': lambda: GymEnvDelayed('Walker2d-v1', delay = DELAY_CONST),
    },
    'humanoid-standup-gym': {
        'default': lambda: GymEnv('HumanoidStandup-v1'),
        'delayed': lambda: GymEnvDelayed('HumanoidStandup-v1', delay = DELAY_CONST),
    },
    'mecs1': {
        'default': lambda: GymEnv('MECS-v1'),
    },
    'mecs2': {
        'default': lambda: GymEnv('MECS-v2'),
    },
    'mecs3': {
        'default': lambda: GymEnv('MECS-v3'),
    },
    'mecs4': {
        'default': lambda: GymEnv('MECS-v4'),
    },
    'mecs5': {
        'default': lambda: GymEnv('MECS-v5'),
    },
    'mecs6': {
        'default': lambda: GymEnv('MECS-v6'),
    },
    'mecs7': {
        'default': lambda: GymEnv('MECS-v7'),
    },
    'mecs8': {
        'default': lambda: GymEnv('MECS-v8'),
    },
    'mecs9': {
        'default': lambda: GymEnv('MECS-v9'),
    },
    'mecs61': {
        'default': lambda: GymEnv('MECS-v61'),
    },
    'mecs10': {
        'default': lambda: GymEnv('MECS-v10'),
    },
    'mecs11': {
        'default': lambda: GymEnv('MECS-v11'),
    },
    'mecs12': {
        'default': lambda: GymEnv('MECS-v12'),
    },
    'mecs13': {
        'default': lambda: GymEnv('MECS-v13'),
    },
    'mecs14': {
        'default': lambda: GymEnv('MECS-v14'),
    },
    'mecs15': {
        'default': lambda: GymEnv('MECS-v15'),
    },
    'mecs16': {
        'default': lambda: GymEnv('MECS-v16'),
    },
    'mecs17': {
        'default': lambda: GymEnv('MECS-v17'),
    },
    'mecs18': {
        'default': lambda: GymEnv('MECS-v18'),
    },
    'mecs19': {
        'default': lambda: GymEnv('MECS-v19'),
    },
    'mecs20': {
        'default': lambda: GymEnv('MECS-v20'),
    },
    'mecs21': {
        'default': lambda: GymEnv('MECS-v21'),
    },
    'mecs22': {
        'default': lambda: GymEnv('MECS-v22'),
    },
    'mecs23': {
        'default': lambda: GymEnv('MECS-v23'),
    },
    'mecs24': {
        'default': lambda: GymEnv('MECS-v24'),
    },
    'mecs25': {
        'default': lambda: GymEnv('MECS-v25'),
    },
    'mecs26': {
        'default': lambda: GymEnv('MECS-v26'),
    },
    'mecs27': {
        'default': lambda: GymEnv('MECS-v27'),
    },
    'mecs28': {
        'default': lambda: GymEnv('MECS-v28'),
    },
    'mecs29': {
        'default': lambda: GymEnv('MECS-v29'),
    },
    'mecs30': {
        'default': lambda: GymEnv('MECS-v30'),
    },
}

ENVS=['mecs1','mecs2','mecs3','mecs4','mecs5','mecs6','mecs7','mecs8','mecs9','mecs61', 'mecs10', 'mecs11','mecs12','mecs13','mecs14','mecs15','mecs16','mecs17','mecs18','mecs19','mecs20','mecs21',
'mecs22','mecs23','mecs24','mecs25','mecs26','mecs27','mecs28','mecs29','mecs30',
'half-cheetah','hopper','ant','walker','humanoid-gym','humanoid-standup-gym','humanoid-rllab']
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_type', type=float, default=10)
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='default')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
                        default='gaussian')
    parser.add_argument('--envn', type=int, default=10)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    args = parser.parse_args()
    args.env = ENVS[args.envn]
    log_dir = '/home/wisrl/Downloads/log_sac/Mujoco/' + args.env
    log_dir += '_c%s'%args.cost_type

    if args.task == 'delayed':
        log_dir += '_delayed%s'%str(DELAY_CONST)
    if 'cross' in args.task:
        log_dir += '_cross'
    log_dir = log_dir + '/SAC'
    if not args.scale==1.0:
        log_dir += '_s%s'%str(args.scale)
    args.log_dir = log_dir

    return args

def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    constants.COST_TYPE = variant['algorithm_params']['cost_type']
    register(
        id='MECS-v1',
        entry_point='sac.envs.environment_V_sweep:MEC_v1',
        max_episode_steps=5000,
    )

    register(
        id='MECS-v2',
        entry_point='sac.envs.env_V_sweep_v2:MEC_v2',
        max_episode_steps=5000,
    )

    register(
        id='MECS-v3',
        entry_point='sac.envs.env_V_sweep_v3:MEC_v3',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v4',
        entry_point='sac.envs.env_V_sweep_v4:MEC_v4',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v5',
        entry_point='sac.envs.env_V_sweep_v5:MEC_v5',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v6',
        entry_point='sac.envs.env_V_sweep_v6:MEC_v6',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v61',
        entry_point='sac.envs.env_V_sweep_v6_with_a:MEC_v6',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v7',
        entry_point='sac.envs.env_V_sweep_v7_new:MEC_v7',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v8',
        entry_point='sac.envs.env_V_sweep_v8_new:MEC_v8',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v9',
        entry_point='sac.envs.env_V_sweep_v9:MEC_v9',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v10',
        entry_point='sac.envs.env_V_sweep_v10:MEC_v10',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v11',
        entry_point='sac.envs.env_V_sweep_v11:MEC_v11',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v12',
        entry_point='sac.envs.env_V_sweep_v12:MEC_v12',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v13',
        entry_point='sac.envs.env_V_sweep_v13:MEC_v13',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v14',
        entry_point='sac.envs.env_V_sweep_v14:MEC_v14',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v15',
        entry_point='sac.envs.env_V_sweep_v15:MEC_v15',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v16',
        entry_point='sac.envs.env_V_sweep_v16:MEC_v16',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v17',
        entry_point='sac.envs.env_V_sweep_v17:MEC_v17',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v18',
        entry_point='sac.envs.env_V_sweep_v18:MEC_v18',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v19',
        entry_point='sac.envs.env_V_sweep_v19:MEC_v19',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v20',
        entry_point='sac.envs.env_V_sweep_v20:MEC_v20',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v21',
        entry_point='sac.envs.env_V_sweep_v21:MEC_v21',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v22',
        entry_point='sac.envs.env_V_sweep_v22:MEC_v22',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v23',
        entry_point='sac.envs.env_V_sweep_v23:MEC_v23',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v24',
        entry_point='sac.envs.env_V_sweep_v24:MEC_v24',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v25',
        entry_point='sac.envs.env_V_sweep_v25:MEC_v25',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v26',
        entry_point='sac.envs.env_V_sweep_v26:MEC_v26',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v27',
        entry_point='sac.envs.env_V_sweep_v27:MEC_v27',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v28',
        entry_point='sac.envs.env_V_sweep_v28:MEC_v28',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v29',
        entry_point='sac.envs.env_V_sweep_v29:MEC_v29',
        max_episode_steps=5000,
    )
    register(
        id='MECS-v30',
        entry_point='sac.envs.env_V_sweep_v30:MEC_v30',
        max_episode_steps=5000,
    )

    env = normalize(ENVIRONMENTS[domain][task](**env_params))

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(**sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
                env_spec=env.spec,
                hidden_layer_sizes=(M,M),
                reparameterize=policy_params['reparameterize'],
                reg=1e-3,
        )
    elif policy_params['type'] == 'lsp':
        nonlinearity = {
            None: None,
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh
        }[policy_params['preprocessing_output_nonlinearity']]

        preprocessing_hidden_sizes = policy_params.get('preprocessing_hidden_sizes')
        if preprocessing_hidden_sizes is not None:
            observations_preprocessor = MLPPreprocessor(
                env_spec=env.spec,
                layer_sizes=preprocessing_hidden_sizes,
                output_nonlinearity=nonlinearity)
        else:
            observations_preprocessor = None

        policy_s_t_layers = policy_params['s_t_layers']
        policy_s_t_units = policy_params['s_t_units']
        s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

        bijector_config = {
            'num_coupling_layers': policy_params['coupling_layers'],
            'translation_hidden_sizes': s_t_hidden_sizes,
            'scale_hidden_sizes': s_t_hidden_sizes,
        }

        policy = LatentSpacePolicy(
            env_spec=env.spec,
            squash=policy_params['squash'],
            bijector_config=bijector_config,
            reparameterize=policy_params['reparameterize'],
            q_function=qf1,
            observations_preprocessor=observations_preprocessor)
    elif policy_params['type'] == 'gmm':
        # reparameterize should always be False if using a GMMPolicy
        policy = GMMPolicy(
            env_spec=env.spec,
            K=policy_params['K'],
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            qf=qf1,
            reg=1e-3,
        )
    else:
        raise NotImplementedError(policy_params['type'])

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=algorithm_params['lr'],
        scale_reward=algorithm_params['scale']*algorithm_params['scale_reward'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=algorithm_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']
        algo_params = variant['algorithm_params']
        variant['algorithm_params']['scale'] = args.scale
        variant['algorithm_params']['cost_type'] = args.cost_type

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl'],
        )


def main():
    args = parse_args()
    print(ENVIRONMENTS)
    domain, task = args.env, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)
    if args.task == 'delayed':
        variant_generator = get_variants_delayed(domain=domain, task=task, policy=args.policy)
    else:
        variant_generator = get_variants(domain=domain, task=task, policy=args.policy)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()
