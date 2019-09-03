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
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    directory = "results/ppo_fixed_len2019-09-02 16:41:42.821829"
    with open("./{}/args.json".format(directory), 'r') as f:
        args = json.load(f)
    ############## environment parameters ##############
    edge_capability = args["edge_capability"]
    cloud_capability =args["cloud_capability"]
    task_rate = args["task_rate"]
    channel = args["channel"]
    applications = tuple(args["applications"])
    number_of_apps = args["number_of_apps"]
    cloud_policy = args["cloud_policy"]
    use_beta =args["use_beta"]
    silence =args["silence"]

    ############## Hyperparameters ##############
    max_timesteps =args["max_timesteps"]
    action_std =args["action_std"]
    K_epochs =args["K_epochs"]
    eps_clip =args["eps_clip"]
    gamma =args["gamma"]
    lr =args["lr"]
    betas =tuple(args["betas"])

    n_episodes =3
    #############################################

    # filename to load model from
    model_file = "./{}/pytorch_models/env3_200_1999_ppo.pth".format(directory)

    # creating environment
    env = environment.Environment_sosam(task_rate, *applications, use_beta=use_beta)
    state = env.init_for_sosam(edge_capability, cloud_capability, channel)
    state_dim = env.state_dim
    action_dim = env.action_dim

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(model_file))

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            # state, reward, done = env.step(action)
            if t%200==0:
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
                import pdb; pdb.set_trace()
                state, cost, done = env.step_together(t, action, cloud_policy, silence=silence)
                print("new_estimated arrival: {}".format(state[:8]))
                print("new_just arrived: {}".format(state[8:16]))
                print("new_queue length: {}".format(state[16:24]))
                print("new_queue explosion: {}".format(state[24:32]))
                if use_beta:
                    print("new_c_estimated arrival: {}".format(state[32:40]))
                    print("new_c_just arrived: {}".format(state[40:48]))
                    print("new_c_queue length: {}".format(state[48:56]))
                    print("new_c_queue explosion: {}".format(state[56:]))
                print("---------------------------------------")
                print("cost:{}, episode reward{}".format(cost, ep_reward))
                print("---------------------------------------")
                import pdb; pdb.set_trace()
            else:
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
