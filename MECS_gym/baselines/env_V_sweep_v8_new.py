# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy as np
import collections

from baselines.servernode_w_appqueue_w_appinfo_cores import ServerNode as Edge
from baselines.servernode_w_totalqueue_cores import ServerNode as Cloud
from baselines.environment import Environment
from baselines.applications import *
from baselines.channels import *
from baselines.constants import *
from baselines.rl_networks.utils import *
import gym
from gym import error, spaces
from gym.utils import seeding

import time

class MEC_v8(gym.Env):
    def __init__(self, task_rate=10, applications=(SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), time_delta=1, use_beta=True, empty_reward=True, cost_type=COST_TYPE, max_episode_steps=5000, time_stamp=0):
        super().__init__()

        self.state_dim= 0
        self.action_dim= 0
        self.clients = dict()
        self.servers = dict()
        self.links = list()
        self.timestamp = time_stamp
        self.silence = True

        self.applications = applications
        self.task_rate = task_rate#/time_delta
        self.reset_info = list()
        self.use_beta = use_beta
        self.empty_reward = empty_reward
        self.cost_type = cost_type
        self.max_episode_steps = max_episode_steps
        self.before_arrival = None

        channel = WIRED

        edge_capability = NUM_EDGE_CORES * NUM_EDGE_SINGLE * GHZ
        cloud_capability = NUM_CLOUD_CORES * NUM_CLOUD_SINGLE * GHZ
        self.reset_info.append((edge_capability, cloud_capability, channel))
        state = self.init_linked_pair(edge_capability, cloud_capability, channel)
        self.obs_dim = state.size - 3
        print(self.obs_dim)
        # self.action_dim = self.action_dim-2

        # import pdb;pdb.set_trace()
        high = np.ones(self.action_dim)
        low = -high
        self.action_space = spaces.Box(low, high)
        self.action_dim = 0

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.clients = dict()
        self.servers = dict()
        self.links = list()
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def init_linked_pair(self, edge_capability, cloud_capability, channel):
        client = self.add_client(edge_capability)
        client.make_application_queues(*self.applications)

        server = self.add_server(cloud_capability)

        self.add_link(client, server, channel)

        # self.reset_info.append((edge_capability, cloud_capability, channel))
        state = self._get_obs()

        self.state_dim = len(state)
        self.action_dim += len(client.get_applications())+1
        if self.use_beta:
            # if the client use several servers later, action_dim should be increased.
            self.action_dim *=2
        return state

    def add_client(self, cap):
        client = Edge(cap, True)
        self.clients[client.get_uuid()] = client
        return client

    def add_server(self, cap):
        server = Cloud(cap)
        self.servers[server.get_uuid()] = server
        return server

    def add_link(self, client, server, up_channel, down_channel=None):
        up_channel = Channel(up_channel)
        if not down_channel:
            down_channel = Channel(up_channel)
        else:
            down_channel = Channel(down_channel)

        client.links_to_higher[server.get_uuid()]= {
            'node' : server,
            'channel' : up_channel
        }
        server.links_to_lower[client.get_uuid()] = {
            'node' : client,
            'channel' : down_channel
        }
        self.links.append((client.get_uuid(), server.get_uuid()))
        return

    def get_number_of_apps(self):
        return len(self.applications)

    def __del__(self):
        for k in list(self.clients.keys()):
            del self.clients[k]
        for k in list(self.servers.keys()):
            del self.servers[k]
        del self.links
        del self.applications
        # del self.reset_info

    def reset(self, empty_reward=True, rand_start = 0):
        task_rate = self.task_rate
        applications = self.applications
        # reset_info = self.reset_info
        use_beta = self.use_beta
        cost_type = self.cost_type
        time_stamp = self.timestamp
        self.__del__()
        self.__init__(task_rate, applications, use_beta = use_beta, empty_reward=empty_reward, cost_type=cost_type, time_stamp=time_stamp)
        for reset_info in self.reset_info:
            self.init_linked_pair(*reset_info)
        self.before_arrival = self.get_edge_qlength(scale=GHZ)
        _, failed_to_generate, _ = self._step_generation()
        reset_state = self._get_obs(scale=GHZ)
        reset_state = reset_state[:-3]
        reset_state[-1]=0.0
        return reset_state

    def _get_obs(self, scale=GHZ):
        edge_state, cloud_state, link_state = list(), list(), list()
        for client in self.clients.values():
            temp_state = client._get_obs(self.timestamp, scale=scale)
            edge_state += temp_state

        state = edge_state
        if self.use_beta:
            for server in self.servers.values():
                temp_state = server._get_obs(self.timestamp, scale=scale)
                cloud_state += temp_state
            for link in self.links:
                link_state.extend([self.clients[link[0]].sample_channel_rate(link[1]),self.servers[link[1]].sample_channel_rate(link[0])])

            state = edge_state + cloud_state
        return np.array(state)

    # several clients, several servers not considered. (step, _step_alpha, _step_beta)
    def step(self, action, use_beta=True, generate=True):
        # print(action)
        q0 = np.array(self.before_arrival)
        action = np.clip(action,-1.0,1.0)
        action = 5 * action
        start_state = self._get_obs(scale=GHZ)
        q1=np.array(self.get_edge_qlength(scale=GHZ))
        action_alpha, action_beta, usage_ratio = list(), list(), list()
        if self.use_beta:
            # action_alpha = np.array(list(action[:int(len(action)/2)]) + [0.0]).reshape(1,-1)
            # action_beta = np.array(list(action[int(len(action)/2):]) + [0.0]).reshape(1,-1)
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
            ### softmax here
            action_beta = softmax_1d(action_beta)
        else:
            action_alpha = action
        ### softmax here
        action_alpha = softmax_1d(action_alpha)
        # action_alpha = np.array([[0,0,0,1]])
        # action_beta = np.array([[0,0,0,1]])

        if self.timestamp%1000==0:
            print("--------------------------------------------")
            print("action", action)
            print("action_alpha", action_alpha)
            print("action_beta", action_beta)
            print("q_length",q1)


        # print("--------------------------------------------")
        # print("action", action)
        # print("action_alpha", action_alpha)
        # print("action_beta", action_beta)
        # print(used_edge_cpus)
        # print(used_cloud_cpus)
        used_edge_cpus, inter_state = self._step_alpha(action_alpha)
        ## print("action_beta", action_beta)
        # if self.timestamp < 500000:
        #     cost_type = 1000
        # else:
        #     cost_type = np.copy(self.cost_type)

        used_cloud_cpus, new_state, q_last = self._step_beta(action_beta)
        self.before_arrival = q_last
        q_increase = np.array(q_last) - q0
        _, failed_to_generate, _ = self._step_generation()

        new_state= self._get_obs(scale=GHZ)
        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, q_increase, cost_type=self.cost_type)
        new_state = new_state[:-3]
        new_state[-1] = list(used_cloud_cpus.values())[0]/GHZ/216
        self.timestamp += 1
        if self.timestamp == self.max_episode_steps:
            return new_state, -cost, 1, {"cloud_cpu_used":start_state[-1]}
        return new_state, -cost, 0, {"cloud_cpu_used":start_state[-1]}

    def _step_alpha(self, action):
        # initial_qlength= self.get_total_qlength()
        used_edge_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)
        state = self._get_obs(scale=GHZ)

        if self.timestamp%1000==0:
            print("alpha", 1-sum(sum(action)))
        return used_edge_cpus, state


    def _step_beta(self, action):
        used_txs = collections.defaultdict(list)
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        # 모든 client 객체에 대해 각 client의 상위 node로 offload하기
        # 각 client는 하나의 상위 노드를 가지고 있다고 가정함......?
        q_before = self.get_edge_qlength(scale=GHZ)
        # print(q_before)
        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node)
                used_txs[higher_node].append(used_tx)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)
        q_last = self.get_edge_qlength(scale=GHZ)
        # print(np.array(q_last)-np.array(q_before))
        # print((np.array(q_last)-np.array(q_before))*np.array([10000,20000,40000]))
        s1 = self._get_obs(scale=GHZ)
        for server_id, server in self.servers.items():
            # print("s1",server.task_queue.get_cpu_needs(scale=1))
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], self.timestamp)
            # print("s2",server.task_queue.get_cpu_needs(scale=1))
            s2 = self._get_obs(scale=GHZ)
            used_cloud_cpus[server_id] = server.do_tasks()
            used_cloud_cpus[server_id] = (s2-s1)[-2]*GHZ
        # print(s1)
        # print(s2)
        # print(GHZ)

        state = self._get_obs(scale=GHZ)
        if self.timestamp%1000==0:
            print("beta", 1-sum(sum(action)))
        return used_cloud_cpus, state, q_last

    def _step_generation(self):
        initial_qlength= self.get_edge_qlength()
        if not self.silence: print("###### random task generation start! ######")
        for client in self.clients.values():
            arrival_size, failed_to_generate = client.random_task_generation(self.task_rate, self.timestamp, *self.applications)
        if not self.silence: print("###### random task generation ends! ######")

        after_qlength = self.get_edge_qlength(scale=GHZ)

        return initial_qlength, failed_to_generate, after_qlength

    def get_edge_qlength(self, scale=1):
        qlengths = list()
        # if self.timestamp % 1000 == 0:
        #     print(self.clients.values())
        for node in self.clients.values():
            for _, queue in node.get_queue_list():
                # if self.timestamp % 1000 == 0:
                #     print(queue.get_length(scale,print1=True))
                qlengths.append( queue.get_length(scale) )
        return qlengths

    def get_cloud_qlength(self, scale=1):
        qlengths = list()
        for node in self.servers.values():
            qlengths.append(node.get_task_queue_length(scale))
        return np.array(qlengths)


    def get_cost(self, used_edge_cpus, used_cloud_cpus, before, increase, cost_type, failed_to_offload=0, failed_to_generate=0):
        def compute_cost_fct(cores, cpu_usage):
            return cores*(cpu_usage/400/GHZ/cores)**3

        # every queue lengths are in unit of  [10e9 bits]
        # q_len_sum_after = sum(np.array(after))
        # q_len_sum_before = sum(np.array(before))



        # edge_drift_cost = (q_len_sum_after-q_len_sum_before)

        edge_drift_cost = sum(before*increase)

        # if self.timestamp%1000==0:
        #     print("q len : ", after)
        #     print("drift : ", edge_drift_cost)
        #
        # if edge_drift_cost>1:
        #     edge_drift_cost=edge_drift_cost-1
        # else:
        #     edge_drift_cost=np.log(np.maximum(edge_drift_cost,1e-20))

        # print(edge_drift_cost)
        if self.timestamp%1000==0:
            print("drift after : ", edge_drift_cost)
        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            edge_computation_cost += compute_cost_fct(10,used_edge_cpu)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            cloud_payment_cost += compute_cost_fct(54,used_cloud_cpu)

        # edge_computation_cost = np.log(np.maximum(edge_computation_cost,1e-30))
        # cloud_payment_cost = np.log(np.maximum(cloud_payment_cost,1e-30))
        # print(edge_computation_cost)
        # print(cloud_payment_cost)
        # print(used_edge_cpus)
        # print(used_cloud_cpus)
        # print(edge_computation_cost)
        # print(cloud_payment_cost)
        if self.timestamp%1000==0:
            print("used cpu edge : ", used_edge_cpus.values())
            print("used cpu cloud : ", used_cloud_cpus.values())
            print("edge power : ", edge_computation_cost)
            print("cloud power : ", cloud_payment_cost)
            print("power * cost : ", cost_type*(edge_computation_cost+cloud_payment_cost))
            print("cost : ", edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost))
            print("rew : ", -(edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost)))

        # # print("asdasdasd")
        # print(used_edge_cpus)
        # print(used_cloud_cpus)
        #
        # print(edge_drift_cost)
        # print("dddd")
        # print(edge_computation_cost)
        # print(cloud_payment_cost)
        # edge_computation_cost=0
        # cloud_payment_cost =0
        return (edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost))
