import os,sys
import pathlib
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)
import torch
import argparse
import json
import numpy as np
from datetime import datetime
from servernode_w_queue import ServerNode
from applications import *
from channels import *
from utilities import *
from constants import *
import environment_ppo as environment
import pickle
from rl.ppo.ppo_fixed_len import PPO
from rl.ppo.ppo_utils import *


def main():

    parser = argparse.ArgumentParser()
    ############## environment parameters ##############
    parser.add_argument('--edge_capability', default = 3.0*1e2*GHZ, metavar='G', help = "total edge CPU capability", type=int)
    parser.add_argument('--cloud_capability', default = 2.5*1e2*GHZ, metavar='G', help = "total cloud CPU capability", type=int)  # clock per tick
    parser.add_argument('--channel', default = WIRED, metavar='G')
    parser.add_argument('--applications', default = (SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), metavar='G')#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
    parser.add_argument('--use_beta', action = 'store_true', help = "use 'offload' to cloud")
    parser.add_argument('--silence', action = 'store_true', help= "shush environment messages")


    ############## Hyperparameters ##############
    parser.add_argument('--log_interval', default = 20 , metavar='N', help="print avg reward in the interval", type=int)
    parser.add_argument('--max_episodes', default = 1000, metavar='N', help="max training episodes", type=int)
    parser.add_argument('--max_timesteps', default = 2000, metavar='N', help="max timesteps in one episode", type=int)

    parser.add_argument('--update_timestep', default = 1000, metavar='N', help="update policy every n timesteps", type=int)
    parser.add_argument('--action_std', default = 0.5 , metavar='N', help="constant std for action distribution (Multivariate Normal)", type=float)
    parser.add_argument('--K_epochs', default = 80  , metavar='N', help="update policy for K epochs")
    parser.add_argument('--eps_clip', default = 0.2 , metavar='N', help="clip parameter for PPO", type=float)
    parser.add_argument('--gamma', default = 0.9   , metavar='N', help="discount factor", type=float)

    parser.add_argument('--lr', default = 0.0003 , metavar='N', help="parameters for Adam optimizer", type=float)
    parser.add_argument('--betas', default = (0.9, 0.999), metavar='N')
    parser.add_argument('--random_seed', default = None, metavar='N')
    #############################################


    ############## save parameters ##############
    file_name = 'ppo_fixed_len'+str(datetime.now())
    result_dir = "./{}/eval_results".format(file_name)
    model_dir = "./{}/pytorch_models".format(file_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args = parser.parse_args()
    with open("./{}/args.json".format(file_name), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    ############## parser arguments to plain variabales ##############
    edge_capability = args.edge_capability
    cloud_capability = args.cloud_capability
    channel = args.channel
    applications = args.applications
    use_beta = args.use_beta
    silence = args.silence
    number_of_apps = len(applications)
    cloud_policy = [1/number_of_apps]*number_of_apps

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

    # creating environment
    env = environment.Environment_sosam(1, *applications, use_beta=use_beta)
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
    evaluations_empty_reward_1000 = []
    evaluations_1000 = []

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, cost, done = env.step_together(time_step, action, cloud_policy, silence=silence)
            reward = -cost
            # Saving reward:
            memory.rewards.append(reward)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

            if done:
                break
            # if t%200==0:
            #     print("episode {}, average length {}, running_reward{}".format(i_episode, avg_length, running_reward))

        avg_length += t
        evaluations_empty_reward.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps*2))
        evaluations.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps*2, empty_reward=False))
        evaluations_empty_reward_1000.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=1000))
        evaluations_1000.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=1000, empty_reward=False))
        np.save("{}/eval_empty_reward".format(result_dir), evaluations_empty_reward)
        np.save("{}/eval".format(result_dir), evaluations)
        np.save("{}/eval_empty_reward_1000".format(result_dir), evaluations_empty_reward_1000)
        np.save("{}/eval_1000".format(result_dir), evaluations_1000)
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        #     break

        # save every 500 episodes
        if i_episode % 50 == 0:
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
