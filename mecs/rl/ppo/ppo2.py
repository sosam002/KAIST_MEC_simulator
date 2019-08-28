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
        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 32)
        self.alpha_action_mean = nn.Linear(32, int(action_dim/2))
        self.beta_action_mean = nn.Linear(32, int(action_dim/2))
        self.action_log_std = nn.Parameter(torch.zeros(1,action_dim))


        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)


    def forward(self, state, memory):
        raise NotImplementedError

    def act(self, state, memory):
        x = F.tanh(self.affine1(state))
        x = F.tanh(self.affine2(x))
        alpha = self.alpha_action_mean(x)
        beta = self.beta_action_mean(x)
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
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy_old.act(state, memory)
        action = F.softmax(action.reshape(2,-1)/2).cpu().data.numpy().flatten()
        return action

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        # 여기서 discounted_reward가 0이므로 td reward인듯
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
            #### 4번째 탭에 dist entropy 를 1로 바꿔봄?
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            loss = -torch.min(surr1, surr2) + 0.01*self.MseLoss(state_values, rewards) - dist_entropy

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
