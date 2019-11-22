import os,sys
import pathlib
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)
import torch
import argparse
import json
import numpy as np
from datetime import datetime
# from servernode_w_queue_appinfo import ServerNode
# from applications import *
# from channels import *
# from utilities import *
from constants import *
import environment_ppo_under1latent_cost1_univ as environment
import pickle
from rl.ppo_fixed_len import PPO
from rl.ppo_utils import *


def main():

    parser = argparse.ArgumentParser()
    ############## environment parameters ##############
    parser.add_argument('--edge_capability', default = 4*1e2*GHZ, metavar='G', help = "total edge CPU capability", type=int)
    parser.add_argument('--cloud_capability', default = 2.4*1e3*GHZ, metavar='G', help = "total cloud CPU capability", type=int)  # clock per tick
    parser.add_argument('--task_rate', default = 10, metavar='G', help = "application arrival task rate")
    parser.add_argument('--channel', default = WIRED, metavar='G')
    parser.add_argument('--applications', default = (SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), metavar='G')#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
    parser.add_argument('--cost_type', default = 0, metavar='G', type=int)
    parser.add_argument('--use_beta', action = 'store_true', help = "use 'offload' to cloud")
    parser.add_argument('--silence', action = 'store_true', help= "shush environment messages")

    parser.add_argument('--comment', default=None)
    parser.add_argument('--save', action = 'store_true')

    ############## Hyperparameters ##############
    parser.add_argument('--log_interval', default = 20 , metavar='N', help="print avg reward in the interval", type=int)
    parser.add_argument('--max_episodes', default = 2000, metavar='N', help="max training episodes", type=int)
    parser.add_argument('--max_timesteps', default = 2000, metavar='N', help="max timesteps in one episode", type=int)

    parser.add_argument('--update_timestep', default = 1000, metavar='N', help="update policy every n timesteps", type=int)
    parser.add_argument('--action_std', default = 0.5 , metavar='N', help="constant std for action distribution (Multivariate Normal)", type=float)
    parser.add_argument('--K_epochs', default = 80  , metavar='N', help="update policy for K epochs")
    parser.add_argument('--eps_clip', default = 0.2 , metavar='N', help="clip parameter for PPO", type=float)
    parser.add_argument('--gamma', default = 0.9   , metavar='N', help="discount factor", type=float)

    parser.add_argument('--lr', default = 0.0003 , metavar='N', help="parameters for Adam optimizer", type=float)
    parser.add_argument('--betas', default = (0.9, 0.999), metavar='N')
    parser.add_argument('--random_seed', default = 1, metavar='N', type=float)
    #############################################

    args = parser.parse_args()
    args_dict = vars(args)

    ############## parser arguments to plain variabales ##############
    edge_capability = args.edge_capability
    cloud_capability = args.cloud_capability
    task_rate = args.task_rate
    channel = args.channel
    applications = args.applications
    cost_type = args.cost_type
    use_beta = args.use_beta
    silence = args.silence
    save = args.save

    number_of_apps = len(applications)
    cloud_policy = [1/number_of_apps]*number_of_apps
    args_dict['number_of_apps'] = number_of_apps
    args_dict['cloud_policy'] = cloud_policy

    log_interval = args.log_interval
    max_episodes = args.max_episodes
    max_timesteps = args.max_timesteps
    update_timestep = args.update_timestep
    action_std = args.action_std
    K_epochs = args.K_epochs
    eps_clip = args.eps_clip
    gamma = args.gamma
    lr = args.lr
    betas = args.betas
    random_seed = args.random_seed
    ##################################################################

    ############## save parameters ##############
    if save:
        file_name = 'ppo_fixed_under1dummy_newnetwork'+str(datetime.now())
        eval_dir = "./results/{}/eval_results".format(file_name)
        model_dir = "./results/{}/pytorch_models".format(file_name)

        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open("./results/{}/args.json".format(file_name), 'w') as f:
            json.dump(args_dict, f, indent='\t')
    # import pdb; pdb.set_trace()
    # creating environment
    env = environment.MEC_v1(task_rate, *applications, use_beta=use_beta, cost_type=cost_type)
    state = env.init_for_sosam(edge_capability, cloud_capability, channel)
    state_dim = env.state_dim
    action_dim = env.action_dim

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    evaluations_empty_reward = []
    evaluations = []
    # evaluations_empty_reward_1000 = []
    # evaluations_1000 = []

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            if t%250==0:
                print("---------------------------------------")
                print("estimated arrival: {}".format(state[:8]))
                print("just arrived: {}".format(state[8:16]))
                print("queue length: {}".format(state[16:24]))
                print("queue explosion: {}".format(state[24:32]))
                if use_beta:
                    print("c_estimated arrival: {}".format(state[32:40]))
                    print("c_just arrived: {}".format(state[40:48]))
                    print("c_queue length: {}".format(state[48:56]))
                    print("c_queue explosion: {}".format(state[56:]))
                print("---------------------------------------")
                print("------action\t{}".format(action))
                print("---------------------------------------")
            state, cost, done = env.step(t, action, cloud_policy, silence=silence)
            if t%250==0:
                print("new_estimated arrival: {}".format(state[:8]))
                print("new_just arrived: {}".format(state[8:16]))
                print("new_queue length: {}".format(state[16:24]))
                print("new_queue explosion: {}".format(state[24:32]))
                if use_beta:
                    print("new_c_estimated arrival: {}".format(state[32:40]))
                    print("new_c_just arrived: {}".format(state[40:48]))
                    print("new_c_queue length: {}".format(state[48:56]))
                    print("new_c_queue explosion: {}".format(state[56:]))

            reward = -cost
            # Saving reward:
            memory.rewards.append(reward)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if t%250==0:
                print("---------------------------------------")
                print("cost:{}, episode reward{}".format(cost, running_reward))
                print("---------------------------------------")
            if done:
                break
            # if t%200==0:
            #     print("episode {}, average length {}, running_reward{}".format(i_episode, avg_length, running_reward))

        avg_length += t
        # import pdb; pdb.set_trace()
        evaluations_empty_reward.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps*2))
        evaluations.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps*2, empty_reward=False))
        # evaluations_empty_reward_1000.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=1000))
        # evaluations_1000.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=1000, empty_reward=False))
        if save:
            np.save("{}/eval_empty_reward".format(eval_dir), evaluations_empty_reward)
            np.save("{}/eval".format(eval_dir), evaluations)
        # np.save("{}/eval_empty_reward_1000".format(eval_dir), evaluations_empty_reward_1000)
        # np.save("{}/eval_1000".format(eval_dir), evaluations_1000)
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        #     break

        # save every 500 episodes
        if save and i_episode % 50 == 0:
            ppo.save('env3_{}_{}'.format(i_episode, t), directory=model_dir)

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
