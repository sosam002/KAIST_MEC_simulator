# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy as np
import collections

from servernode_w_appqueue_w_appinfo_cores import ServerNode as Edge
from servernode_w_totalqueue_cores import ServerNode as Cloud
from environment import Environment
from applications import *
from channels import *
from constants import *
from cost_functions_cores import *

class MEC_v1(Environment):
    def __init__(self, task_rate, *applications, time_delta=1, use_beta=True, empty_reward=True, cost_type=1, max_episode_steps=5000):
        super().__init__()
        self.applications = applications
        self.task_rate = task_rate#/time_delta
        self.reset_info = list()
        self.use_beta = use_beta
        self.empty_reward = empty_reward
        self.cost_type = cost_type
        self.max_episode_steps = max_episode_steps

    def init_linked_pair(self, edge_capability, cloud_capability, channel):
        client = self.add_client(edge_capability)
        client.make_application_queues(*self.applications)

        server = self.add_server(cloud_capability)
        # server.make_application_queues(*self.applications)

        self.add_link(client, server, channel)

        self.reset_info.append((edge_capability, cloud_capability, channel))
        # state,_,_ = self.get_status()
        state = self.get_status()
        self.state_dim = len(state)
        # for client in self.clients.values():
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

        # client = self.clients[client_num]
        # server = self.servers[server_num]
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
        del self.reset_info

    def reset(self, empty_reward=True, rand_start = 0):
        task_rate = self.task_rate
        applications = self.applications
        reset_info = self.reset_info
        use_beta = self.use_beta
        cost_type = self.cost_type
        self.__del__()
        self.__init__(task_rate, *applications, use_beta = use_beta, empty_reward=empty_reward, cost_type=cost_type)
        for reset_info in reset_info:
            self.init_linked_pair(*reset_info)
        # reset_state,_,_ = self.get_status()
        reset_state = self.get_status(scale=GHZ)
        return reset_state

    def get_status(self, scale=GHZ):
        edge_state, cloud_state, link_state = list(), list(), list()
        # failed_to_offload, failed_to_generate = 0, 0
        for client in self.clients.values():
            temp_state = client.get_status(self.timestamp, scale=scale)
            edge_state += temp_state
            # failed_to_generate += sum(temp_state[24:32])

        state = edge_state
        if self.use_beta:
            for server in self.servers.values():
                temp_state = server.get_status(self.timestamp, scale=scale, cpu_ver=True)
                cloud_state += temp_state
                # failed_to_offload +=sum(temp_state[24:32])
            for link in self.links:
                link_state.extend([self.clients[link[0]].sample_channel_rate(link[1]),self.servers[link[1]].sample_channel_rate(link[0])])

            state = edge_state + cloud_state

        # client이면서 server인 경우 state를 한 번만 표시하도록 바꿔야 함

        return np.array(state)#, failed_to_offload, failed_to_generate

    # several clients, several servers not considered. (step, _step_alpha, _step_beta)
    # def step(self, action, cloud, use_beta=True, generate=True):
    def step(self, action, use_beta=True, generate=True):
        q0, failed_to_generate, q1 = self._step_generation()
        action_alpha, action_beta, usage_ratio = list(), list(), list()
        if self.use_beta:
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
        else:
            action_alpha = action

        used_edge_cpus, inter_state, q2 = self._step_alpha(action_alpha)
        # used_cloud_cpus, new_state, failed_to_offload, q3 = self._step_beta(action_beta, np.array(cloud).reshape(-1,len(cloud)))
        # used_cloud_cpus, new_state = self._step_beta(action_beta, np.array(cloud).reshape(-1,len(cloud)))
        used_cloud_cpus, new_state = self._step_beta(action_beta)
        # fail_cost = self.get_fail_cost(failed_to_offload, failed_to_generate)

        # cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, q3, failed_to_offload, failed_to_generate)
        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, q2)
        self.timestamp += 1
        if self.timestamp == self.max_episode_steps:
            return new_state, cost, 1
        # return new_state, cost, failed_to_offload+failed_to_generate
        return new_state, cost, 0

    def _step_alpha(self, action):
        # initial_qlength= self.get_total_qlength()
        used_edge_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        if self.timestamp%1000==0:
            print("alpha", 1-sum(sum(action)))
        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)

        # state, _, _ = self.get_status()
        state = self.get_status(scale=GHZ)
        # after_qlength = self.get_total_qlength()
        after_qlength = self.get_edge_qlength(scale=GHZ)

        return used_edge_cpus, state, after_qlength


    # def _step_beta(self, action, action_cloud):
    def _step_beta(self, action):
        used_txs = collections.defaultdict(list)
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        if self.timestamp%1000==0:
            print("beta", 1-sum(sum(action)))
        # 모든 client 객체에 대해 각 client의 상위 node로 offload하기
        # 각 client는 하나의 상위 노드를 가지고 있다고 가정함......?
        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                # import pdb; pdb.set_trace()/
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node)
                used_txs[higher_node].append(used_tx)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)

        # server_action = dict(zip(self.servers.keys(), action_cloud))
        for server_id, server in self.servers.items():
            # import pdb; pdb.set_trace()
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], self.timestamp)
            used_cloud_cpus[server_id] = server.do_tasks()

        # after_qlength = self.get_total_qlength()
        # state, failed_to_offload, _ = self.get_status()
        state = self.get_status(scale=GHZ)
        # return used_cloud_cpus, state, failed_to_offload, after_qlength
        return used_cloud_cpus, state

    def _step_generation(self):
        # initial_qlength= self.get_total_qlength()
        initial_qlength= self.get_edge_qlength()
        if not self.silence: print("###### random task generation start! ######")
        for client in self.clients.values():
            arrival_size, failed_to_generate = client.random_task_generation(self.task_rate, self.timestamp, *self.applications)
        if not self.silence: print("###### random task generation ends! ######")
        # after_qlength = self.get_total_qlength()
        after_qlength = self.get_edge_qlength(scale=GHZ)

        return initial_qlength, failed_to_generate, after_qlength

    # def get_total_qlength(self, scale):
    #     qlength = list()
    #     for node in self.clients.values():
    #         for _, queue in node.get_queue_list():
    #             qlength.append( queue.get_length(scale) )
    #     if self.use_beta:
    #         for node in self.servers.values():
    #             for _, queue in node.get_queue_list():
    #                 qlength.append( queue.get_length(scale) )
    #     return qlength

    def get_edge_qlength(self, scale=1):
        qlengths = list()
        for node in self.clients.values():
            for _, queue in node.get_queue_list():
                qlengths.append( queue.get_length(scale) )
        return qlengths

    def get_cloud_qlength(self, scale=1):
        qlengths = list()
        for node in self.servers.values():
            qlengths.append(node.get_task_queue_length(scale,cpu_ver=True))
        return np.array(qlengths)
    def get_cost(self, used_edge_cpus, used_cloud_cpus, before, after, failed_to_offload=0, failed_to_generate=0):

        # import pdb; pdb.set_trace()
        # for 3 apps
        edge_drift_cost = sum(np.array(after)**2)
        # for 8 apps
        # edge_drift_cost = 10*sum(np.array(after)**2)

        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            if self.cost_type==10 or self.cost_type==11 or self.cost_type==14 or self.cost_type==16 or self.cost_type>99:
                edge_computation_cost += 10*(used_edge_cpu/10)**3
            else: # 12, 13, 15
                edge_computation_cost += get_local_power_cost(used_edge_cpu)
        edge_computation_cost /= (900000*64*GHZ**3)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            # for 3 apps
            cloud_payment_cost += 54*(used_cloud_cpu/54)**3
            # for 8 apps
            # cloud_payment_cost += 45*(used_cloud_cpu/45)**3
        cloud_payment_cost /= (900000*64*GHZ**3)


        if self.cost_type==10 or self.cost_type==12:
            return 100*edge_drift_cost + 1*edge_computation_cost + 50*cloud_payment_cost
        elif self.cost_type==11 or self.cost_type==13:
            return 100*edge_drift_cost + 50*edge_computation_cost + 1*cloud_payment_cost
        elif self.cost_type==14 or self.cost_type==15:
            return 100*edge_drift_cost + 100*edge_computation_cost + 1*cloud_payment_cost
        elif self.cost_type==16 or self.cost_type==17:
            return 100*edge_drift_cost + 100*edge_computation_cost + 0.1*cloud_payment_cost
        elif self.cost_type>99:
            return self.cost_type*edge_drift_cost/100 + 10*edge_computation_cost + 10*cloud_payment_cost
        else:
            return (10**self.cost_type)*edge_drift_cost/100 + 10*edge_computation_cost + 10*cloud_payment_cost
