import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from ppo_utils import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.affine1 = nn.Linear(state_dim, 128)
        self.affine2 = nn.Linear(128, 64)
        self.alpha_action_mean = nn.Linear(64, int(action_dim/2))
        self.beta_action_mean = nn.Linear(64, int(action_dim/2))
        self.action_log_std = nn.Parameter(torch.zeros(1,action_dim))


        self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                )

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)


    def forward(self, state, memory):
        raise NotImplementedError

    def act(self, state, memory):
        x = F.tanh(self.affine1(state))
        x = F.tanh(self.affine2(x))
        alpha = self.alpha_action_mean(x)
        beta = self.beta_action_mean(x)
        # import pdb; pdb.set_trace()
        action_mean = torch.cat((alpha,beta),dim=1)

        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        # action = F.softmax(action.reshape(2,-1)).reshape(1,-1)
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        # import pdb; pdb.set_trace()
        x = F.tanh(self.affine1(state))
        x = F.tanh(self.affine2(x))
        alpha = self.alpha_action_mean(x)
        beta = self.beta_action_mean(x)
        action_mean = torch.cat((alpha,beta),dim=1)
        # action_mean = torch.squeeze(x)

        action_var = self.action_var.expand_as(action_mean) #action_log_std
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # action_logprobs = dist.log_prob(torch.squeeze(action))
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        # import pdb; pdb.set_trace()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, c1=0.01, c2=1):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)

        self.MseLoss = nn.MSELoss()

        self.c1 = c1
        self.c2 = c2

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy_old.act(state, memory)
        action = F.softmax(action.reshape(2,-1)/2).cpu().data.numpy().flatten()
        return action

    def update(self, memory, c1=0.01, c2=1):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            loss = -torch.min(surr1, surr2) + self.c1*self.MseLoss(state_values, rewards) - self.c2*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filename, directory):
        torch.save(self.policy.state_dict(), '{}/{}_ppo.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load('{}/{}_ppo.pth'.format(directory, filename)))

#
# import json
# import torch
# import logging
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
#
# import shutil
# import copy
# from datetime import datetime
#
# # from mecs
# import scheduler, config
# # from wholemap import WholeMap
#
#
# from servernode_w_queue import ServerNode
#
# from applications import *
# from channels import *
# from utilities import *
# from constants import *
# import environment3_ppo as environment
# import pickle
#
# def evaluate_policy(env, policy, cloud_policy, memory, epsd_length=1000, eval_episodes=10):
#     print("---------------------------------------")
#     print("EVALUATION STARTED")
#     print("---------------------------------------")
#     eval_mem = Memory()
#     # import pdb; pdb.set_trace()
#     eval_mem.actions = memory.actions[-epsd_length:]
#     eval_mem.states = memory.states[-epsd_length:]
#     eval_mem.logprobs = memory.logprobs[-epsd_length:]
#     eval_mem.rewards = memory.rewards[-epsd_length:]
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#     	obs = env.reset()
#     	done = False
#     	costs=[]
#
#     	for t in range(epsd_length):
#     		silence=True
#     		if t%200==0:
#     			silence=False
#
#     		action = policy.select_action(obs, eval_mem)
#     		# import pdb; pdb.set_trace()
#     		obs, cost, failed = env.step(t, np.array(action).reshape(-1,len(action)), cloud_policy, silence=silence)
#
#     		avg_reward -= cost
#     		costs.append(cost)
#
#     		if failed or t==999:
#     			print("episode length {}".format(t))
#     			break
#
#     avg_reward /= eval_episodes
#
#     print("---------------------------------------")
#     print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
#     print("---------------------------------------")
#     return avg_reward
#
#
# def main():
#
#     ############## save parameters ##############
#     file_name = 'ppo2_fixed_len'+str(datetime.now())
#     result_dir = "./results/{}".format(file_name)
#     model_dir = "./pytorch_models/{}".format(file_name)
#
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#
#
#     ############## environment parameters ##############
#
#     # 4번째 탭이 2.5, 2.5 3번째 탭이 3.0 2.5
#     edge_capability = 3.0*1e2*GHZ
#     cloud_capability = 2.5*1e2*GHZ  # clock per tick
#     channel = WIRED
#     applications = SPEECH_RECOGNITION, NLP, FACE_RECOGNITION#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
#     number_of_apps = len(applications)
#     cloud_policy = [1/number_of_apps]*number_of_apps
#     use_beta = True
#     silence = True
#
#
#     ############## Hyperparameters ##############
#     render = False
#     # solved_reward = 300         # stop training if avg_reward > solved_reward
#     log_interval = 20           # print avg reward in the interval
#     max_episodes = 10000        # max training episodes
#     max_timesteps = 2000        # max timesteps in one episode
#
#     update_timestep = 1000      # update policy every n timesteps
#     action_std = 0.5            # constant std for action distribution (Multivariate Normal)
#     K_epochs = 80               # update policy for K epochs
#     eps_clip = 0.1              # clip parameter for PPO (3,4번째는 0.2, 다섯번째는 0.1로 줄였음)
#     gamma = 0.9                # discount factor
#
#     lr = 0.0003                 # parameters for Adam optimizer
#     betas = (0.9, 0.999)
#
#     random_seed = None
#     #############################################
#
#     # creating environment
#     env = environment.MEC_v1(1, *applications, use_beta=use_beta)
#     state = env.init_for_sosam(edge_capability, cloud_capability, channel)
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#
#     if random_seed:
#         print("Random Seed: {}".format(random_seed))
#         torch.manual_seed(random_seed)
#         # env.seed(random_seed)
#         np.random.seed(random_seed)
#
#     memory = Memory()
#     ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
#     print(lr,betas)
#
#     # logging variables
#     running_reward = 0
#     avg_length = 0
#     time_step = 0
#     evaluations = []
#     evaluations_fixed_len = []
#
#     # training loop
#     for i_episode in range(1, max_episodes+1):
#         state = env.reset()
#         for t in range(max_timesteps):
#             time_step +=1
#             # Running policy_old:
#             action = ppo.select_action(state, memory)
#             state, cost, done = env.step(time_step, action, cloud_policy, silence=silence)
#             reward = -cost
#             # Saving reward:
#             memory.rewards.append(reward)
#
#             # update if its time
#             if time_step % update_timestep == 0:
#                 ppo.update(memory)
#                 memory.clear_memory()
#                 time_step = 0
#             running_reward += reward
#
#             if done:
#                 break
#             if t%200==0:
#                 print("episode {}, average length {}, running_reward{}".format(i_episode, avg_length, running_reward))
#
#         avg_length += t
#         evaluations.append(evaluate_policy(env, ppo, cloud_policy, memory, epsd_length=max_timesteps))
#         np.save("{}/eval".format(result_dir), evaluations)
#         # stop training if avg_reward > solved_reward
#         # if running_reward > (log_interval*solved_reward):
#         #     print("########## Solved! ##########")
#         #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
#         #     break
#
#         # save every 500 episodes
#         if i_episode % 200 == 0:
#             ppo.save('env3_{}_{}'.format(i_episode, t), directory=model_dir)
#
#         # logging
#         if i_episode % log_interval == 0:
#             avg_length = int(avg_length/log_interval)
#             running_reward = int((running_reward/log_interval))
#
#             print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
#             running_reward = 0
#             avg_length = 0
#
# if __name__ == '__main__':
#     main()
