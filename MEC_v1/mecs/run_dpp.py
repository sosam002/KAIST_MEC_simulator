import os,sys
import pathlib
import argparse
import json
import numpy as np
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)

from constants import *
import environment_dpp as environment
import pickle
import json
from results_list import dir_names
from datetime import datetime
from dpp import KKT_actor
import time


def main():

    parser = argparse.ArgumentParser()
    ############## environment parameters ##############
    parser.add_argument('--edge_cores', default = 10, metavar='G', help = "total edge CPU capability", type=float)  # clock per tick, unit=GHZ
    parser.add_argument('--edge_single', default = 4, metavar='G', help = "total edge CPU capability", type=float)  # clock per tick, unit=GHZ
    parser.add_argument('--cloud_cores', default = 54, metavar='G', help = "total cloud CPU capability", type=float)  # clock per tick, unit=GHZ
    parser.add_argument('--cloud_single', default = 4, metavar='G', help = "total cloud CPU capability", type=float)  # clock per tick, unit=GHZ
    parser.add_argument('--task_rate', default = 10, metavar='G', help = "application arrival task rate")
    parser.add_argument('--channel', default = WIRED, metavar='G')
    parser.add_argument('--applications', default = (SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), metavar='G')#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
    parser.add_argument('--cost_type', default = 0, metavar='G', type=float)
    parser.add_argument('--use_beta', action = 'store_true', help = "use 'offload' to cloud")
    parser.add_argument('--silence', action = 'store_true', help= "shush environment messages")

    parser.add_argument('--comment', default=None)
    parser.add_argument('--save', action = 'store_true')

    ############## Hyperparameters ##############
    parser.add_argument('--max_episodes', default = 10000, metavar='N', help="max training episodes", type=int)
    # horizen ~5000
    parser.add_argument('--max_episode_steps', default = 5000, metavar='N', help="max timesteps in one episode", type=int)
    parser.add_argument('--random_seed', default = 1, metavar='N', type=int)
    parser.add_argument('--iter', default = 5, type=int)
    parser.add_argument('--scale', default = 1, type=float)

    #############################################

    args = parser.parse_args()
    args_dict = vars(args)

    ############## parser arguments to plain variabales ##############
    edge_cores = args.edge_cores
    edge_single = args.edge_single
    cloud_cores = args.cloud_cores
    cloud_single = args.cloud_single
    task_rate = args.task_rate
    channel = args.channel
    applications = args.applications
    cost_type = args.cost_type
    use_beta = args.use_beta
    silence = args.silence
    save = args.save
    iter=args.iter
    scale=args.scale

    number_of_apps = len(applications)
    args_dict['number_of_apps'] = number_of_apps

    max_episodes = args.max_episodes
    max_episode_steps = args.max_episode_steps
    random_seed = args.random_seed
    ##################################################################

    ############## save parameters ##############
    if save:
        file_name = './dppresults/c{}_s{}_d{}'.format(cost_type, scale, str(datetime.now()))
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        with open("{}/args.json".format(file_name), 'w') as f:
            json.dump(args_dict, f, indent='\t')
        simul_dir = "{}/simulate".format(file_name)
        if not os.path.exists(simul_dir):
            os.makedirs(simul_dir)

    # creating environment
    env1 = environment.MEC_v1(task_rate, *applications, use_beta=use_beta, cost_type=cost_type, max_episode_steps=max_episode_steps)
    state = env1.init_linked_pair(edge_cores*edge_single*GHZ, cloud_cores*cloud_single*GHZ, channel)
    state_dim = env1.state_dim
    action_dim = env1.action_dim

    actor = KKT_actor(edge_cores*edge_single*GHZ, cloud_cores*cloud_single*GHZ, 0.02*GBPS, cost_type, scale)
    ep_rewards = []

    for ep in range(iter):
        states = []
        actions = []
        running_times = []
        state = env1.reset()
        ep_reward = 0
        for t in range(max_episode_steps):
            # Running policy_old:

            start_time = time.time()
            action = actor.optimize(state)
            end_time=time.time()
            running_times.append(end_time-start_time)
            states.append(state)
            actions.append(action)

            action = actor.optimize(state)

            state, cost, done = env1.step(action)

            reward = -cost
            ep_reward += reward

            if done:
                break

        if save:
            np.save("{}/states_{}".format(simul_dir, ep), states)
            np.save("{}/actions_{}".format(simul_dir, ep), actions)
            np.save("{}/runtime_{}".format(simul_dir,ep), running_times)

        print("runtime{}".format(np.array(running_times).mean()))
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_rewards.append(ep_reward)
        ep_reward = 0
        # env1.close()
    if save:
        eval_dir = "{}/eval_results".format(file_name)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        np.save("{}/ep_reward".format(eval_dir), ep_rewards)
if __name__ == '__main__':
    main()
