import numpy as np

from rllab.misc.instrument import VariantGenerator
from sac.misc.utils import flatten, get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

LSP_POLICY_PARAMS_BASE = {
    'type': 'lsp',
    'coupling_layers': 2,
    's_t_layers': 1,
    'action_prior': 'uniform',
    # 'preprocessing_hidden_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',
    'reparameterize': REPARAMETERIZE,
    'squash': True
}

LSP_POLICY_PARAMS = {
    'swimmer-gym': { # 2 DoF
        'preprocessing_hidden_sizes': (M, M, 4),
        's_t_units': 2,
    },
    'swimmer-rllab': { # 2 DoF
        'preprocessing_hidden_sizes': (M, M, 4),
        's_t_units': 2,
    },
    'hopper': { # 3 DoF
        'preprocessing_hidden_sizes': (M, M, 6),
        's_t_units': 3,
    },
    'half-cheetah': { # 6 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'walker': { # 6 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
   },
    'ant': { # 8 DoF
        'preprocessing_hidden_sizes': (M, M, 16),
        's_t_units': 8,
    },
    'humanoid-gym': { # 17 DoF
        'preprocessing_hidden_sizes': (M, M, 34),
        's_t_units': 17,
    },
    'humanoid-standup-gym': {  # 17 DoF
        'preprocessing_hidden_sizes': (M, M, 34),
        's_t_units': 17,
    },
    'humanoid-rllab': { # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 42),
        's_t_units': 21,
    },

    'mecs1': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 16),
        's_t_units': 8,
    },
    'mecs2': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs3': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs4': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs5': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs6': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs61': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs7': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs8': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs9': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs10': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs11': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs12': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs13': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs14': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs15': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs16': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs17': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs18': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs19': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs20': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs21': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs22': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs23': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs24': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs25': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs26': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs27': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs28': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs29': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
    'mecs30': {  # 21 DoF
        'preprocessing_hidden_sizes': (M, M, 12),
        's_t_units': 6,
    },
}

GMM_POLICY_PARAMS_BASE = {
    'type': 'gmm',
    'K': 1,
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

GMM_POLICY_PARAMS = {
    'swimmer-gym': { # 2 DoF
    },
    'swimmer-rllab': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-rllab': { # 21 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    'mecs1': { # 17 DoF
    },
    'mecs2': { # 17 DoF
    },
    'mecs3': { # 17 DoF
    },
    'mecs4': { # 17 DoF
    },
    'mecs5': { # 17 DoF
    },
    'mecs6': { # 17 DoF
    },
    'mecs61': { # 17 DoF
    },
    'mecs7': { # 17 DoF
    },
    'mecs8': { # 17 DoF
    },
    'mecs9': { # 17 DoF
    },
    'mecs10': { # 17 DoF
    },
    'mecs11': { # 17 DoF
    },
    'mecs12': { # 17 DoF
    },
    'mecs13': { # 17 DoF
    },
    'mecs14': { # 17 DoF
    },
    'mecs15': { # 17 DoF
    },
    'mecs16': { # 17 DoF
    },
    'mecs17': { # 17 DoF
    },
    'mecs18': { # 17 DoF
    },
    'mecs19': { # 17 DoF
    },
    'mecs20': { # 17 DoF
    },
    'mecs21': { # 17 DoF
    },
    'mecs22': { # 17 DoF
    },
    'mecs23': { # 17 DoF
    },
    'mecs24': { # 17 DoF
    },
    'mecs25': { # 17 DoF
    },
    'mecs26': { # 17 DoF
    },
    'mecs27': { # 17 DoF
    },
    'mecs28': { # 17 DoF
    },
    'mecs29': { # 17 DoF
    },
    'mecs30': { # 17 DoF
    },
}

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'gaussian',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

GAUSSIAN_POLICY_PARAMS = {
    'swimmer-gym': { # 2 DoF
    },
    'swimmer-rllab': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-rllab': { # 21 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    'mecs1': { # 17 DoF
    },
    'mecs2': { # 17 DoF
    },
    'mecs3': { # 17 DoF
    },
    'mecs4': { # 17 DoF
    },
    'mecs5': { # 17 DoF
    },
    'mecs6': { # 17 DoF
    },
    'mecs61': { # 17 DoF
    },
    'mecs7': { # 17 DoF
    },
    'mecs8': { # 17 DoF
    },
    'mecs9': { # 17 DoF
    },
    'mecs10': { # 17 DoF
    },
    'mecs11': { # 17 DoF
    },
    'mecs12': { # 17 DoF
    },
    'mecs13': { # 17 DoF
    },
    'mecs14': { # 17 DoF
    },
    'mecs15': { # 17 DoF
    },
    'mecs16': { # 17 DoF
    },
    'mecs17': { # 17 DoF
    },
    'mecs18': { # 17 DoF
    },
    'mecs19': { # 17 DoF
    },
    'mecs20': { # 17 DoF
    },
    'mecs21': { # 17 DoF
    },
    'mecs22': { # 17 DoF
    },
    'mecs23': { # 17 DoF
    },
    'mecs24': { # 17 DoF
    },
    'mecs25': { # 17 DoF
    },
    'mecs26': { # 17 DoF
    },
    'mecs27': { # 17 DoF
    },
    'mecs28': { # 17 DoF
    },
    'mecs29': { # 17 DoF
    },
    'mecs30': { # 17 DoF
    },
}

POLICY_PARAMS = {
    'lsp': {
        k: dict(LSP_POLICY_PARAMS_BASE, **v)
        for k, v in LSP_POLICY_PARAMS.items()
    },
    'gmm': {
        k: dict(GMM_POLICY_PARAMS_BASE, **v)
        for k, v in GMM_POLICY_PARAMS.items()
    },
    'gaussian': {
        k: dict(GAUSSIAN_POLICY_PARAMS_BASE, **v)
        for k, v in GAUSSIAN_POLICY_PARAMS.items()
    },
}

VALUE_FUNCTION_PARAMS = {
    'layer_size': M,
}

ENV_DOMAIN_PARAMS = {
    'swimmer-gym': { # 2 DoF
    },
    'swimmer-rllab': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-rllab': { # 21 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    'mecs1': { # 17 DoF
    },
    'mecs2': { # 17 DoF
    },
    'mecs3': { # 17 DoF
    },
    'mecs4': { # 17 DoF
    },
    'mecs5': { # 17 DoF
    },
    'mecs6': { # 17 DoF
    },
    'mecs61': { # 17 DoF
    },
    'mecs7': { # 17 DoF
    },
    'mecs8': { # 17 DoF
    },
    'mecs9': { # 17 DoF
    },
    'mecs10': { # 17 DoF
    },
    'mecs11': { # 17 DoF
    },
    'mecs12': { # 17 DoF
    },
    'mecs13': { # 17 DoF
    },
    'mecs14': { # 17 DoF
    },
    'mecs15': { # 17 DoF
    },
    'mecs16': { # 17 DoF
    },
    'mecs17': { # 17 DoF
    },
    'mecs18': { # 17 DoF
    },
    'mecs19': { # 17 DoF
    },
    'mecs20': { # 17 DoF
    },
    'mecs21': { # 17 DoF
    },
    'mecs22': { # 17 DoF
    },
    'mecs23': { # 17 DoF
    },
    'mecs24': { # 17 DoF
    },
    'mecs25': { # 17 DoF
    },
    'mecs26': { # 17 DoF
    },
    'mecs27': { # 17 DoF
    },
    'mecs28': { # 17 DoF
    },
    'mecs29': { # 17 DoF
    },
    'mecs30': { # 17 DoF
    },
}

ENV_PARAMS = {
    'swimmer-gym': { # 2 DoF
    },
    'swimmer-rllab': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
        'resume-training': {
            'low_level_policy_path': [
                # 'ant-low-level-policy-00-00/itr_4000.pkl',
            ]
        },
        'cross-maze': {
            'terminate_at_goal': True,
            'goal_reward_weight': [1000],
            'goal_radius': 2,
            'velocity_reward_weight': 0,
            'ctrl_cost_coeff': 0, # 1e-2,
            'contact_cost_coeff': 0, # 1e-3,
            'survive_reward': 0, # 5e-2,
            'goal_distance': 12,
            'goal_angle_range': (0, 2*np.pi),
            'reward_type' : 'sparse',

            # 'env_fixed_goal_position': [[6, -6], [6, 6], [12, 0]],
            #
            # 'pre_trained_policy_path': []
        },
    },
    'humanoid-gym': { # 17 DoF
        'resume-training': {
            'low_level_policy_path': [
                # 'humanoid-low-level-policy-00-00/itr_4000.pkl',
            ]
        }
    },
    'humanoid-rllab': { # 21 DOF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    'mecs1': { # 17 DoF
    },
    'mecs2': { # 17 DoF
    },
    'mecs3': { # 17 DoF
    },
    'mecs4': { # 17 DoF
    },
    'mecs5': { # 17 DoF
    },
    'mecs6': { # 17 DoF
    },
    'mecs61': { # 17 DoF
    },
    'mecs7': { # 17 DoF
    },
    'mecs8': { # 17 DoF
    },
    'mecs9': { # 17 DoF
    },
    'mecs10': { # 17 DoF
    },
    'mecs11': { # 17 DoF
    },
    'mecs12': { # 17 DoF
    },
    'mecs13': { # 17 DoF
    },
    'mecs14': { # 17 DoF
    },
    'mecs15': { # 17 DoF
    },
    'mecs16': { # 17 DoF
    },
    'mecs17': { # 17 DoF
    },
    'mecs18': { # 17 DoF
    },
    'mecs19': { # 17 DoF
    },
    'mecs20': { # 17 DoF
    },
    'mecs21': { # 17 DoF
    },
    'mecs22': { # 17 DoF
    },
    'mecs23': { # 17 DoF
    },
    'mecs24': { # 17 DoF
    },
    'mecs25': { # 17 DoF
    },
    'mecs26': { # 17 DoF
    },
    'mecs27': { # 17 DoF
    },
    'mecs28': { # 17 DoF
    },
    'mecs29': { # 17 DoF
    },
    'mecs30': { # 17 DoF
    },
}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': 0.999,
    'target_update_interval': 1,
    'tau': 0.005,
    'reparameterize': REPARAMETERIZE,

    'base_kwargs': {
        'epoch_length': 5000,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': 5000,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'swimmer-gym': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'swimmer-rllab': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'hopper': { # 3 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'ant': { # 8 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'mecs1': { # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs2': { # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs3': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs4': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs5': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs6': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs61': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs7': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs8': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs9': {  # 8 DoF
        'scale_reward': 1,
        'discount': 1.0,
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 50000,
        }
    },
    'humanoid-gym': { # 17 DoF
        'scale_reward': 20,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-rllab': { # 21 DoF
        'scale_reward': 10,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-standup-gym': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'mecs10': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs11': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs12': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs13': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs14': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs15': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs16': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs17': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs18': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs19': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs20': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs21': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs22': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs23': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs24': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs25': {  # 8 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs26': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs27': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs28': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs29': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
    'mecs30': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
}

ALGORITHM_PARAMS_DELAYED = {
    'swimmer-gym': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'swimmer-rllab': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'hopper': { # 3 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'ant': { # 8 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'humanoid-gym': { # 17 DoF
        'scale_reward': 20,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-rllab': { # 21 DoF
        'scale_reward': 10,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-standup-gym': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
}

REPLAY_BUFFER_PARAMS = {
    'max_replay_buffer_size': 1e6,
}

SAMPLER_PARAMS = {
    'max_path_length': 5000,
    'min_pool_size': 5000,
    'batch_size': 256,
}

RUN_PARAMS_BASE = {
    'seed': [1],
    'snapshot_mode': 'gap',
    'snapshot_gap': 5000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'swimmer-gym': { # 2 DoF
        'snapshot_gap': 50000
    },
    'swimmer-rllab': { # 2 DoF
        'snapshot_gap': 50000
    },
    'hopper': { # 3 DoF
        'snapshot_gap': 50000
    },
    'half-cheetah': { # 6 DoF
        'snapshot_gap': 50000
    },
    'walker': { # 6 DoF
        'snapshot_gap': 50000
    },
    'ant': { # 8 DoF
        'snapshot_gap': 50000
    },
    'humanoid-gym': { # 21 DoF
        'snapshot_gap': 50000
    },
    'humanoid-standup-gym': {  # 21 DoF
        'snapshot_gap': 50000
    },
    'humanoid-rllab': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs1': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs2': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs3': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs4': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs5': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs6': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs61': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs7': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs8': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs9': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs10': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs11': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs12': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs13': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs14': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs15': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs16': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs17': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs18': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs19': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs20': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs21': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs22': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs23': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs24': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs25': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs26': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs27': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs28': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs29': { # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs30': { # 21 DoF
        'snapshot_gap': 50000
    },
}


DOMAINS = [
    'swimmer-gym', # 2 DoF
    'swimmer-rllab', # 2 DoF
    'hopper', # 3 DoF
    'half-cheetah', # 6 DoF
    'walker', # 6 DoF
    'ant', # 8 DoF
    'humanoid-gym', # 17 DoF # gym_humanoid
    'humanoid-rllab', # 21 DoF
    'humanoid-standup-gym', # 17 DoF # gym_humanoid
    'mecs1', # 17 DoF # gym_humanoid
    'mecs2', # 17 DoF # gym_humanoid
    'mecs3', # 17 DoF # gym_humanoid
    'mecs4', # 17 DoF # gym_humanoid
    'mecs5', # 17 DoF # gym_humanoid
    'mecs6', # 17 DoF # gym_humanoid
    'mecs61', # 17 DoF # gym_humanoid
    'mecs7', # 17 DoF # gym_humanoid
    'mecs8', # 17 DoF # gym_humanoid
    'mecs9', # 17 DoF # gym_humanoid
    'mecs10', # 17 DoF # gym_humanoid
    'mecs11', # 17 DoF # gym_humanoid
    'mecs12', # 17 DoF # gym_humanoid
    'mecs13', # 17 DoF # gym_humanoid
    'mecs14', # 17 DoF # gym_humanoid
    'mecs15', # 17 DoF # gym_humanoid
    'mecs16', # 17 DoF # gym_humanoid
    'mecs17', # 17 DoF # gym_humanoid
    'mecs18', # 17 DoF # gym_humanoid
    'mecs19', # 17 DoF # gym_humanoid
    'mecs20', # 17 DoF # gym_humanoid
    'mecs21', # 17 DoF # gym_humanoid
    'mecs22', # 17 DoF # gym_humanoid
    'mecs23', # 17 DoF # gym_humanoid
    'mecs24', # 17 DoF # gym_humanoid
    'mecs25', # 17 DoF # gym_humanoid
    'mecs26', # 17 DoF # gym_humanoid
    'mecs27', # 17 DoF # gym_humanoid
    'mecs28', # 17 DoF # gym_humanoid
    'mecs29', # 17 DoF # gym_humanoid
    'mecs30', # 17 DoF # gym_humanoid
]

TASKS = {
    'swimmer-gym': [
        'default',
        'delayed',
    ],
    'swimmer-rllab': [
        'default',
        'multi-direction',
    ],
    'hopper': [
        'default',
        'delayed',
    ],
    'half-cheetah': [
        'default',
        'delayed',
    ],
    'walker': [
        'default',
        'delayed',
    ],
    'ant': [
        'default',
        'multi-direction',
        'cross-maze',
        'delayed',

    ],
    'humanoid-gym': [
        'default',
        'delayed',
    ],
    'humanoid-rllab': [
        'default',
        'multi-direction'
    ],
    'humanoid-standup-gym': [
        'default',
        'delayed',
    ],
    'mecs1': [
        'default',
    ],
    'mecs2': [
        'default',
    ],
    'mecs3': [
        'default',
    ],
    'mecs4': [
        'default',
    ],
    'mecs5': [
        'default',
    ],
    'mecs6': [
        'default',
    ],
    'mecs61': [
        'default',
    ],
    'mecs7': [
        'default',
    ],
    'mecs8': [
        'default',
    ],
    'mecs9': [
        'default',
    ],
    'mecs10': [
        'default',
    ],
    'mecs11': [
        'default',
    ],
    'mecs12': [
        'default',
    ],
    'mecs13': [
        'default',
    ],
    'mecs14': [
        'default',
    ],
    'mecs15': [
        'default',
    ],
    'mecs16': [
        'default',
    ],
    'mecs17': [
        'default',
    ],
    'mecs18': [
        'default',
    ],
    'mecs19': [
        'default',
    ],
    'mecs20': [
        'default',
    ],
    'mecs21': [
        'default',
    ],
    'mecs22': [
        'default',
    ],
    'mecs23': [
        'default',
    ],
    'mecs24': [
        'default',
    ],
    'mecs25': [
        'default',
    ],
    'mecs26': [
        'default',
    ],
    'mecs27': [
        'default',
    ],
    'mecs28': [
        'default',
    ],
    'mecs29': [
        'default',
    ],
    'mecs30': [
        'default',
    ],
}


def parse_domain_and_task(env_name):
    domain = next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TASKS[domain]
    task = next((task for task in domain_tasks if task in env_name), 'default')
    return domain, task

def get_variants(domain, task, policy):
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domain]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def get_variants_delayed(domain, task, policy):
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_DELAYED[domain]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg
