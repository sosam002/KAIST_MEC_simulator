import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, action_dim*10)
		self.l2 = nn.Linear(action_dim*10, action_dim*4)
		self.l3_alpha = nn.Linear(action_dim*4, int(action_dim*0.5))
		self.l3_beta = nn.Linear(action_dim*4, int(action_dim*0.5))

		self.max_action = max_action


	def forward(self, x):
		# import pdb; pdb.set_trace()
		# print(self.l1.weight)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		alpha = F.softmax(self.l3_alpha(x))
		beta = F.softmax(self.l3_beta(x))
		return torch.cat((alpha,beta), dim=1)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, action_dim*10)
		self.l2 = nn.Linear(action_dim*10, action_dim*4)
		self.l3 = nn.Linear(action_dim*4, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, action_dim*10)
		self.l5 = nn.Linear(action_dim*10, action_dim*4)
		self.l6 = nn.Linear(action_dim*4, 1)


	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1


class TD3(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.action_dim = action_dim
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		if np.isnan(self.actor(state).cpu().data.numpy().flatten()[0]):
			import pdb; pdb.set_trace()
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.02, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Select action according to policy and add clipped noise
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			# import pdb; pdb.set_trace()
			noise = abs(noise.clamp(-noise_clip, noise_clip))
			# noise = noise.clamp(-noise_clip, noise_clip)
			next_action = F.softmax((self.actor_target(next_state) + noise).reshape(-1,int(self.action_dim/2))).reshape(-1, self.action_dim)
			# next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				# import pdb; pdb.set_trace()
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

				# Optimize the actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
