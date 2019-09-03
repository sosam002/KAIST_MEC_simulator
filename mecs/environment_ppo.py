# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy as np
import collections

# from mecs.mobilenode import MobileNode
# from mecs.servernode import ServerNode
from servernode_w_queue import ServerNode

from applications import *
from channels import *
from utilities import *
from constants import *

class Environment_sosam:
    def __init__(self, task_rate, *applications, time_delta=10*MS, use_beta=False, empty_reward=True):
        self.task_rate = task_rate#/time_delta
        self.clients = {}
        self.servers = {}
        self.links = collections.OrderedDict()
        self.applications = applications
        self.reset_infos = []
        self.state_dim=0
        self.action_dim=0
        self.use_beta = use_beta
        self.max_episode_steps = 4000
        self.empty_reward = empty_reward

    def get_number_of_apps(self):
        return len(self.applications)

    def __del__(self):
        for k in list(self.clients.keys()):
            del self.clients[k]
        for k in list(self.servers.keys()):
            del self.servers[k]
        del self.links
        del self.applications
        del self.reset_infos

    def reset(self, empty_reward=True):

        task_rate = self.task_rate
        applications = self.applications
        reset_infos = self.reset_infos
        use_beta = self.use_beta
        self.__del__()
        self.__init__(task_rate, *applications, use_beta = use_beta, empty_reward=empty_reward)
        for reset_info in reset_infos:
            self.init_for_sosam(*reset_info)
        reset_state,_,_ = self.get_status(0)
        return reset_state

    def add_client(self, cap):
        client = ServerNode(cap, True)
        self.clients[client.get_uuid()] = client
        return client

    def add_server(self, cap):
        server = ServerNode(cap)
        self.servers[server.get_uuid()] = server
        return server

    def add_link(self, client, server, up_channel, down_channel=None):
        if not down_channel:
            down_channel = up_channel

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
        self.links[(client.get_uuid(), server.get_uuid())] = [down_channel, up_channel]
        return


    def init_for_sosam(self, edge_capability, cloud_capability, channel):
        client = self.add_client(edge_capability)
        client.make_application_queues(*self.applications)

        if self.use_beta:
            server = self.add_server(cloud_capability)
            ##
            # self.add_link(0, 0, channel)
            # for client in self.clients:
            #     for server in self.servers:
            self.add_link(client, server, channel)

            server.make_application_queues(*self.applications)

            # self.clients[0].make_application_queues(*self.applications)
            # self.servers[0].make_application_queues(*self.applications)
        self.reset_infos.append((edge_capability, cloud_capability, channel))
        state,_,_ = self.get_status(0)
        self.state_dim = len(state)
        for client in self.clients.values():
            self.action_dim += len(client.get_applications())
        if self.use_beta:
            self.action_dim *=2
        return state

    def get_action(self):
        action_alpha = abs(np.random.normal(0, 0.5, size=int(self.action_dim/2)))
        action_beta = abs(np.random.normal(0, 0.5, size=int(self.action_dim/2)))
        return action_alpha/np.sum(action_alpha), action_beta/np.sum(action_beta)


    def print_queue_lists(self):
        for node in self.clients.values():
            for app_type, queue_list in node.get_queue_list():
                print("client server {} tasks {}".format(app_type, queue_list.get_tasks()))
        if self.use_beta:
            for node in self.servers.values():
                for app_type, queue_list in node.get_queue_list():
                    print("cloud server {} tasks {}".format(app_type, queue_list.get_tasks()))

        return

    def Lyap_function2(self, normalize=1):
        lyap = 0
        for node in self.clients.values():
            for _, queue in node.get_queue_list():
                lyap += queue.length/queue.max_length*normalize
        if self.use_beta:
            for node in self.servers.values():
                for _, queue in node.get_queue_list():
                    lyap += queue.length/queue.max_length*normalize
        return lyap

    def step_together(self, time, action, cloud, use_beta=True, generate=True, silence=True):
        q0, failed_to_generate, q1 = self._step_generation(time, silence)
        action_alpha, action_beta = list(), list()
        if self.use_beta:
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
        else:
            action_alpha = action

        used_edge_cpus, inter_state, q2 = self._step_alpha(time, action_alpha)
        used_cloud_cpus, new_state, failed_to_offload, q3 = self._step_beta(time, action_beta, np.array(cloud).reshape(-1,len(cloud)))
        fail_cost = self.get_fail_cost(time, failed_to_offload, failed_to_generate)
        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, self.get_drift_cost(q0, q3)) + fail_cost

        return new_state, cost, failed_to_offload+failed_to_generate

    def _step_generation(self, time, silence=True):
        initial_Lyap= self.Lyap_function2()
        if not silence: print("###### random task generation start! ######")
        for client in self.clients.values():
            arrival_size, failed_to_generate = client.random_task_generation(self.task_rate, time, *self.applications)
        if not silence: print("###### random task generation ends! ######")
        after_Lyap = self.Lyap_function2()

        return initial_Lyap, failed_to_generate, after_Lyap

    def _step_alpha(self, time, action, silence =True):
        initial_Lyap= self.Lyap_function2()
        used_edge_cpus = collections.defaultdict(float)


        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)

        state, _, _ = self.get_status(time)
        after_Lyap = self.Lyap_function2()
        # lyap_drift = self.get_drift_cost(initial_Lyap, after_Lyap)
        return used_edge_cpus, state, after_Lyap


    def _step_beta(self, time, action, action_cloud, silence =True):
        used_txs = collections.defaultdict(list)
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)

        # 모든 client 객체에 대해 각 client의 상위 node로 offload하기
        # 각 client는 하나의 상위 노드를 가지고 있다고 가정함......?
        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node)
                used_txs[higher_node].append(used_tx)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)

        server_action = dict(zip(self.servers.keys(), action_cloud))
        for server_id, server in self.servers.items():
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], time)
            used_cloud_cpus[server_id] = server.do_tasks(server_action[server_id])

        after_Lyap = self.Lyap_function2()
        state, failed_to_offload, _ = self.get_status(time)
        return used_cloud_cpus, state, failed_to_offload, after_Lyap

    def get_status(self,time):

        edge_state, cloud_state, link_state = list(), list(), list()
        failed_to_offload, failed_to_generate = 0, 0
        for client in self.clients.values():
            temp_state = client.get_status(time)
            edge_state += temp_state
            failed_to_generate += sum(temp_state[-8:])
        # edge_state = self.clients[0].get_status()
            # print("edge state (queue length+cpu cap.) = {}".format(edge_state))

        # state = estimated_arrival_rate + edge_state
        state = edge_state

        if self.use_beta:
            for server in self.servers.values():
                temp_state = server.get_status(time)
                cloud_state += temp_state
                failed_to_offload +=sum(temp_state[-8:])
            for _, link in self.links.items():
                link_state.append(get_channel_info(link[0])/GBPS)

            state = edge_state + cloud_state

        return np.array(state), failed_to_offload, failed_to_generate

    def get_drift_cost(self, before, after):
        drift = after - before
        if drift > 0:
            drift = after
        if self.empty_reward:
            if after==0:
                drift=-1
        return drift

    def get_cost(self, used_edge_cpus, used_cloud_cpus, drift_cost, silence = True):
        local_cost = 0
        server_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            local_cost += local_energy_consumption(used_edge_cpu)
        for used_cloud_cpu in used_cloud_cpus.values():
            server_cost += local_energy_consumption(used_cloud_cpu)

        cost = my_cost(local_cost, server_cost, drift_cost)

        if not silence: print("cost = {}".format(cost))
        return cost

    def get_fail_cost(self, time, failed_to_offload, failed_to_generate):
         return float((failed_to_offload+failed_to_generate)>0)*self.action_dim
