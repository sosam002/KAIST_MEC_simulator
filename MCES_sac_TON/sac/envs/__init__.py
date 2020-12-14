from gym.envs.registration import register

# 
# register(
#     id='MagellanAnt-v2',
#     entry_point='sac.envs.ant_maze:MagellanAntEnv',
#     max_episode_steps=400
# )
#
# register(
#     id='MagellanHalfCheetahRun-v2',
#     entry_point='sac.envs.half_run:MagellanHalfCheetahRunEnv',
#     max_episode_steps=100
# )
#
# register(
#     id='MagellanHalfCheetahFlip-v2',
#     entry_point='sac.envs.half_flip:MagellanHalfCheetahFlipEnv',
#     max_episode_steps=100
# )
#
# register(
#     id='MagellanSparseMountainCar-v0',
#     entry_point='sac.envs.mountain_car:MagellanSparseContinuousMountainCarEnv',
#     max_episode_steps=500
# )
#
# register(
#     id='SparseHalfCheetah-v1',
#     entry_point='sac.envs.sparse_halfcheetah:SparseHalfCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
#
# register(
#     id='SparseHopper-v1',
#     entry_point='sac.envs.sparse_hopper:SparseHopperEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )
#
# register(
#     id='SparseHopper2-v1',
#     entry_point='sac.envs.sparse_hopper2:SparseHopper2Env',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )
#
# register(
#     id='SparseSwimmer-v1',
#     entry_point='sac.envs.sparse_swimmer:SparseSwimmerEnv',
#     max_episode_steps=1000,
#     reward_threshold=360.0,
# )
#
# register(
#     id='SparseWalker2d-v1',
#     max_episode_steps=1000,
#     entry_point='sac.envs.sparse_walker2d:SparseWalker2dEnv',
# )
#
# register(
#     id='SparseAnt-v1',
#     entry_point='sac.envs.sparse_ant:SparseAntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
#     id='SparseHumanoid-v1',
#     entry_point='sac.envs.sparse_humanoid:SparseHumanoidEnv',
#     max_episode_steps=1000,
# )

from .gym_env import GymEnv, GymEnvDelayed
from .multi_direction_env import (
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv)

from .random_goal_ant_env import RandomGoalAntEnv
from .cross_maze_ant_env import CrossMazeAntEnv
from .hierarchy_proxy_env import HierarchyProxyEnv
from .multigoal import MultiGoalEnv
