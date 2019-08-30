from PIL import Image
import torch
import os, sys, pathlib
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)
from rl.ppo.ppo_fixed_len import PPO, Memory
from servernode_w_queue import ServerNode
from applications import *
from channels import *
from utilities import *
from constants import *
import environment_ppo as environment
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():

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
    n_episodes = 3          # num of episodes to run
    max_timesteps = 4000    # max timesteps in one episode
    # render = True           # render the environment
    # save_gif = False        # png images are saved in gif folder

    # filename and directory to load model from
    filename = "env3_350_1999_ppo.pth"
    directory = "pytorch_models/ppo_fixed_len2019-08-29 16:03:13.198024/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    # creating environment
    env = environment.Environment_sosam(1, *applications, use_beta=use_beta)
    state = env.init_for_sosam(edge_capability, cloud_capability, channel)
    state_dim = env.state_dim
    action_dim = env.action_dim

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            # state, reward, done = env.step(action)
            if t%200==0:
                state, cost, done = env.step_together(t, action, cloud_policy, silence=False)
            state, cost, done = env.step_together(t, action, cloud_policy, silence=silence)
            reward = -cost
            ep_reward += reward
            # if render:
            #     env.render()
            # if save_gif:
            #      img = env.render(mode = 'rgb_array')
            #      img = Image.fromarray(img)
            #      img.save('./gif/{}.jpg'.format(t))
            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        # env.close()

if __name__ == '__main__':
    test()
