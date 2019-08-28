import os,sys
import pathlib
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)
import torch
import numpy as np
from datetime import datetime
from servernode_w_queue import ServerNode
from applications import *
from channels import *
from utilities import *
from constants import *
import environment3_ppo as environment
import pickle
from rl.ppo.ppo_fixed_len import PPO
from rl.ppo.ppo_utils import *


def main():

    ############## save parameters ##############
    file_name = 'ppo2_fixed_len'+str(datetime.now())
    result_dir = "./results/{}".format(file_name)
    model_dir = "./pytorch_models/{}".format(file_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    ############## environment parameters ##############

    # 4번째 탭이 2.5, 2.5 3번째 탭이 3.0 2.5
    edge_capability = 3.0*1e2*GHZ
    cloud_capability = 2.5*1e2*GHZ  # clock per tick
    channel = WIRED
    applications = SPEECH_RECOGNITION, NLP, FACE_RECOGNITION#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
    number_of_apps = len(applications)
    cloud_policy = [1/number_of_apps]*number_of_apps
    use_beta = True
    silence = True


    ############## Hyperparameters ##############
    render = False
    # solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 2000        # max timesteps in one episode

    update_timestep = 1000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO (3,4번째는 0.2, 다섯번째는 0.1로 줄였음)
    gamma = 0.9                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

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
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    evaluations = []
    evaluations_fixed_len = []

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
            if t%200==0:
                print("episode {}, average length {}, running_reward{}".format(i_episode, avg_length, running_reward))

        avg_length += t
        evaluations.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps))
        np.save("{}/eval".format(result_dir), evaluations)
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        #     break

        # save every 500 episodes
        if i_episode % 200 == 0:
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
