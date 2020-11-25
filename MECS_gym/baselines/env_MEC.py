# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy as np
import collections

from baselines.servernode_w_appqueue_w_appinfo_cores_0 import ServerNode as NormalNode
from baselines.servernode_w_totalqueue_cores_0 import ServerNode as IntegNode
from baselines.environment import Environment
from baselines.applications import *
from baselines.channels import *
from baselines.constants import *
from baselines.rl_networks.utils import *
import gym
from gym import error, spaces
from gym.utils import seeding

import time

class MEC_v0(gym.Env):
    def __init__(self, task_rate=10, applications=(SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), channel = WIRED, use_offload=True, offload_type = 'partial', cost_type=COST_TYPE, max_episode_steps=5000, time_stamp=0):
        super().__init__()

        self.action_dim= 0
        self.clients = dict()
        self.servers = dict()
        self.links = list()
        self.timestamp = time_stamp
        self.silence = True

        self.offload_type = offload_type
        self.applications = applications
        self.task_rate = task_rate
        self.reset_info = list()
        self.use_offload = use_offload
        self.cost_type = cost_type
        self.max_episode_steps = max_episode_steps


        self.reset_info.append((NUM_EDGE_CORES, EDGE_SINGLE_CLK, NUM_CLOUD_CORES, CLOUD_SINGLE_CLK, channel))
        state = self.init_linked_pair(NUM_EDGE_CORES, EDGE_SINGLE_CLK, NUM_CLOUD_CORES, CLOUD_SINGLE_CLK, channel)

        self.obs_dim = state.size
        # self.action_dim = self.action_dim-2

        # import pdb;pdb.set_trace()
        high = np.ones(self.action_dim)
        low = -high
        self.action_space = spaces.Box(low, high)
        # self.action_dim = 0

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def init_linked_pair(self, c_cores, c_clk, s_cores, s_clk, up_channel, down_channel=None, c_random_task_generating=True):
        client = self.add_client(c_cores, c_clk, c_random_task_generating)
        server = self.add_server(s_cores, s_clk)
        self.add_link(client, server, up_channel, down_channel)

        client.make_application_queues(*self.applications)


        # self.reset_info.append((edge_capability, cloud_capability, channel))
        state = self._get_obs()

        self.action_dim += len(self.applications)+1
        if self.use_offload:
            # if the client use several servers later, action_dim should be increased.
            self.action_dim *=2
        return state

    def add_client(self, cores, clk, is_random_task_generating):
        client = NormalNode(cores, clk, is_random_task_generating)
        self.clients[client.get_uuid()] = client
        return client

    def add_server(self, cores, clk):
        server = IntegNode(cores, clk)
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

    def _del(self):
        for k in list(self.clients.keys()):
            del self.clients[k]
        for k in list(self.servers.keys()):
            del self.servers[k]
        del self.links
        del self.applications

    def reset(self, rand_start = 0):
        task_rate = self.task_rate
        applications = self.applications
        use_offload = self.use_offload
        cost_type = self.cost_type
        time_stamp = self.timestamp
        self._del()

        self.__init__(task_rate, applications, use_offload = use_offload, cost_type=cost_type, time_stamp=time_stamp)
        _, failed_to_generate, _ = self._step_generation()
        reset_state = self._get_obs(scale=GHZ)
        # reset_state = reset_state[:-3]
        # reset_state[-1] = 0.0
        return reset_state

    def _get_obs(self, scale=GHZ):
        edge_state, cloud_state, link_state = list(), list(), list()
        for client in self.clients.values():
            temp_state = client._get_obs(self.timestamp, scale=scale)
            edge_state += temp_state

        state = edge_state

        # cloud 정보를 쓰지 않음!
        # if self.use_offload:
        #     for server in self.servers.values():
        #         temp_state = server._get_obs(self.timestamp, scale=scale)
        #         cloud_state += temp_state
        #     for link in self.links:
        #         link_state.extend([self.clients[link[0]].sample_channel_rate(link[1]),self.servers[link[1]].sample_channel_rate(link[0])])
        #
        #     state = edge_state + cloud_state
        return np.array(state)

    # several clients, several servers not considered. (step, _step_alpha, _step_beta)
    def step(self, action, use_offload=True, generate=True):

        self._step_move()
        q1=np.array(self.get_edge_qlength(scale=GHZ))

        ############################### RL network actkon part. ####
        action = np.clip(action,-1.0,1.0)
        action = 10 * action
        # 원래 여기 있던 클라우드 씨피유 사용되는 비율이 인포로 들어갔었. 이건 이전 스테이트거 아닌가????

        action_alpha, action_beta = list(), list()
        if self.use_offload:
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
            ### softmax here
            action_beta = softmax_1d(action_beta)
        else:
            action_alpha = action
        ### softmax here
        action_alpha = softmax_1d(action_alpha)
        ##########################################################3

        used_edge_cpus = self._step_alpha(action_alpha)
        used_cloud_cpus = self._step_beta(action_beta)

        # 여기서 comp cap 나누는 하드코딩을 안하려면 알고리즘에서 클라우드 페이 공식에서 숫자 216 고려해서 바꾸면 됨.
        # 서버가 여러개인 상황에서는.. 리스트나 딕셔너리로 넣어야 하는데 그런 경우에도 바꾸기. 지금은 하나인 걸 알고 있으니까 투박하게 다 더함..
        temp_req_cloud_cpus = dict()
        for client in self.clients.values():
            temp_req_cloud_cpus.update(client.get_req_cpus_offloaded(scale=1))

        q_last=np.array(self.get_edge_qlength(scale=GHZ))

        _, failed_to_generate, _ = self._step_generation()

        # 원래 스테이트에서는 비율로 들어가서 /216 했었음. 지금은 216 곱해진 값
        new_state = self._get_obs(scale=GHZ)
        cost = self.get_cost(used_edge_cpus, temp_req_cloud_cpus, q1, q_last, cost_type=self.cost_type)
        self.timestamp += 1
        cloud_info = list(used_cloud_cpus.values())[0]/GHZ/216

        if self.timestamp%1000==0:
            print("--------------------------------------------")
            print("action", action)
            print("action_alpha", action_alpha)
            print("action_beta", action_beta)
            print("state",new_state)
            print("real cloud used", cloud_info*GHZ*216)
            print("required cloud", temp_req_cloud_cpus)

        if self.timestamp == self.max_episode_steps:
            return new_state, -cost, 1, {"cloud_cpu_used":cloud_info}
        return new_state, -cost, 0, {"cloud_cpu_used":cloud_info}

    def _step_move(self):
        for client in self.clients.values():
            if client.movable:
                client.move()
        for server in self.servers.values():
            if server.movable:
                server.move()

    def _step_alpha(self, action):
        # initial_qlength= self.get_total_qlength()
        used_edge_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)
        return used_edge_cpus


    def _step_beta(self, action):
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)

        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node, self.offload_type)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)
        for server_id, server in self.servers.items():
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], self.timestamp)
            used_cloud_cpus[server_id] = server.do_tasks()

        return used_cloud_cpus

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


    def get_cost(self, used_edge_cpus, used_cloud_cpus, before, after, cost_type, failed_to_offload=0, failed_to_generate=0):
        def compute_cost_fct(cores, cpu_usage):
            return cores*(cpu_usage/400/GHZ/cores)**3

        # every queue lengths are in unit of  [10e9 bits]
        q_len_sum_after = sum(np.array(after))
        q_len_sum_before = sum(np.array(before))


        edge_drift_cost = (q_len_sum_after-q_len_sum_before)

        if self.timestamp%1000==0:
            print("drift after : ", edge_drift_cost)
        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            edge_computation_cost += compute_cost_fct(10,used_edge_cpu)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            cloud_payment_cost += compute_cost_fct(54,used_cloud_cpu)

        if self.timestamp%1000==0:
            print("used cpu edge : ", used_edge_cpus.values())
            print("used cpu cloud : ", used_cloud_cpus.values())
            print("edge power : ", edge_computation_cost)
            print("cloud power : ", cloud_payment_cost)
            print("power * cost : ", cost_type*(edge_computation_cost+cloud_payment_cost))
            print("cost : ", edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost))
            print("rew : ", -(edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost)))

        return (edge_drift_cost+cost_type*(edge_computation_cost+cloud_payment_cost))
