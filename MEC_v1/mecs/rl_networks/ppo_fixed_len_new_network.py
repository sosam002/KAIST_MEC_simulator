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
        self.affine_edge1 = nn.Linear(5, 2)
        self.affine_cloud1= nn.Linear(5, 2)
        self.affine_edge2 = nn.Linear(16,32)
        self.affine_cloud2= nn.Linear(16,32)
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

    def _reshape_state(self, state):
        #(state 갯수, edge+cloud=2, 각 node별로 5가지 정보, for all apps)
        state = state.reshape(len(state),2,5,-1)
        state = np.transpose(state,(0,1,3,2))
        return state
    def _action_mean(self, state):
        state = self._reshape_state(state)
        edge_state = F.tanh(self.affine_edge1(state[:,0,:,:]))
        cloud_state = F.tanh(self.affine_edge1(state[:,1,:,:]))
        edge_state = F.tanh(self.affine_edge2(edge_state.reshape(len(edge_state),1,-1)))
        cloud_state = F.tanh(self.affine_edge2(cloud_state.reshape(len(cloud_state),1,-1)))
        aggr_state = torch.cat((edge_state,cloud_state),dim=2).reshape(len(state),-1)
        alpha = self.alpha_action_mean(aggr_state)
        beta = self.beta_action_mean(aggr_state)
        return torch.cat((alpha,beta),dim=1)

    def act(self, state, memory):
        action_mean = self._action_mean(state)

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
        action_mean = self._action_mean(state)
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
