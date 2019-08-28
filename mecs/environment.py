# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy
import collections

# from mecs.mobilenode import MobileNode
# from mecs.servernode import ServerNode
from servernode_w_queue import ServerNode

from applications import *
from channels import *
from utilities import *
from constants import *

logger = logging.getLogger(__name__)
_parent = pathlib.Path(__file__).parent


class Environment_sosam:
    def __init__(self, task_rate, *applications, time_delta=10*MS, use_beta=False):
        self.task_rate = 10#/time_delta
        self.clients = []
        self.servers = []
        self.links = collections.OrderedDict()
        self.applications = applications
        self.reset_infos = []
        self.state_dim=0
        self.action_dim=0
        self.use_beta = use_beta

    def get_number_of_apps(self):
        return len(self.applications)

    def __del__(self):
        while self.clients:
            del self.clients[0]
        while self.servers:
            del self.servers[0]
        del self.links
        del self.applications
        del self.reset_infos

    def reset(self):

        task_rate = self.task_rate
        applications = self.applications
        reset_infos = self.reset_infos
        self.__del__()
        self.__init__(task_rate, *applications)
        for reset_info in reset_infos:
            self.init_for_sosam(*reset_info)
        return self.get_status(0)

    def add_client(self, cap):
        client = ServerNode(cap, True)
        self.clients.append(client)
        return client

    def add_server(self, cap):
        server = ServerNode(cap)
        self.servers.append(server)
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
        # 짜증...ㅜㅜ

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
        state = self.get_status(0)
        self.state_dim = len(state)
        if self.use_beta:
            self.action_dim = len(self.applications)*2
        else:
            self.action_dim = len(self.applications)
        return state

    def get_action_space(self):
        pass

    def print_queue_lists(self):
        for node in self.clients:
            for app_type, queue_list in node.queue_list.items():
                print("client server {} tasks {}".format(app_type, queue_list.tasks))
        if self.use_beta:
            for node in self.servers:
                for app_type, queue_list in node.queue_list.items():
                    print("cloud server {} tasks {}".format(app_type, queue_list.tasks))

        return

    def Lyap_function(self, normalize=100):
        lyap = 0
        for node in self.clients:
            for _, queue in node.queue_list.items():
                lyap += (queue.length/queue.max_length*100)**2
        if self.use_beta:
            for node in self.servers:
                for _, queue in node.queue_list.items():
                    lyap += (queue.length/queue.max_length*100)**2
        return lyap


    def step(self, action, action_cloud, time, generate_random_task=True, silence =True):
        ######################################################################################## 이게 모조리 step 함수
        action_alpha, action_beta = list(), list()
        if self.use_beta:
            action_alpha = action[:self.action_dim/2]
            action_beta = action[self.action_dim/2:]
        else:
            action_alpha = action

        initial_Lyap= self.Lyap_function()

        # perform action (simulation)
        failed_to_generate=0
        if generate_random_task:
            if not silence: print("###### random task generation start! ######")
            arrival_size, failed_to_generate = self.clients[0].random_task_generation(self.task_rate, time, *self.applications)
            if not silence: print("###### random task generation ends! ######")
            # 이건 진짜 arrival rate 이 아님.. arrival만 저장하는걸 또 따로 만들어야 한다니 고통스럽다.
            # print("random task arrival size {}".format(arrival_size))

        used_edge_cpus, used_txs, tasks_to_be_offloaded = [],[],[]
        failed_to_offload = 0
        for client, alpha in list(zip(self.clients, action_alpha)):
            used_edge_cpus.append(client.do_tasks(alpha))
        if self.use_beta:
            for client, beta in list(zip(self.clients, action_beta)):
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, self.servers[0].get_uuid())
                used_txs.append(used_tx)

                tasks_to_be_offloaded.append(task_to_be_offloaded)
                failed_to_offload += sum(failed.values())
            # print("do task on edge, CPU used {}".format(used_edge_cpu))
            # used_tx, task_to_be_offloaded = self.clients[0].offload_tasks(action_beta, self.servers[0].get_uuid())
            # print("offload task to cloud, used_tx {}, offloaded task {}".format(used_tx, task_to_be_offloaded))
            # failed_to_offload = self.servers[0].offloaded_tasks(task_to_be_offloaded, time)

            for task_to_be_offloaded in tasks_to_be_offloaded:
                failed_to_offload += self.servers[0].offloaded_tasks(task_to_be_offloaded, time)
            used_cloud_cpu = self.servers[0].do_tasks(action_cloud[0])


        # print("do task on cloud, CPU used {}".format(used_cloud_cpu))
        after_Lyap= self.Lyap_function()

        state = self.get_status(time)
        cost = self.get_cost(used_edge_cpus, tasks_to_be_offloaded, after_Lyap-initial_Lyap, failed_to_offload, failed_to_generate)


        return state, cost

    def get_status(self,time):
        # estimated_arrival_rate = list()
        # for client in self.clients:
        #     estimated_arrival_rate += list(self.clients[0].estimate_arrival_rate(time)/MB)
        # print("estimated_arrival_rate_state = {}".format(estimated_arrival_rate))
        edge_state, cloud_state, link_state = list(), list(), list()
        for client in self.clients:
            edge_state += client.get_status(time)
        # edge_state = self.clients[0].get_status()
            # print("edge state (queue length+cpu cap.) = {}".format(edge_state))

        # state = estimated_arrival_rate + edge_state
        state = edge_state

        if self.use_beta:
            for server in self.servers:
                cloud_state += server.get_status(time)
            # cloud_state = self.servers[0].get_status()
            # print("cloud state (queue length+cpu cap.) = {}".format(cloud_state))
            # state = estimated_arrival_rate + edge_state + cloud_state + [self.clients[0].get_channel_rate(self.servers[0].get_uuid())/1000000000]
            for _, link in self.links.items():
                link_state.append(get_channel_info(link[0])/GBPS)
            # state = estimated_arrival_rate + edge_state + cloud_state + [self.clients[0].get_channel_rate(self.servers[0].get_uuid())]
            state = edge_state + cloud_state# + link_state

        # print("state :{}".format(state))
            # print("states + channel rate = {}".format(state))
            # state = np.log(np.array(state).clip(1))
        return state

    def get_cost(self, used_edge_cpus, tasks_to_be_offloaded, quad_drift, failed_to_offload, failed_to_generate, silence = True):
        local_cost, server_cost =  0, 0

        for used_edge_cpu in used_edge_cpus:
            local_cost += local_energy_consumption(used_edge_cpu)
        if self.use_beta:
            for task_to_be_offloaded in tasks_to_be_offloaded:
                server_cost += offload_cost(task_to_be_offloaded)
        # cost = my_cost(local_cost, server_cost, quad_drift)*1e-20 + 100*(failed_to_offload+failed_to_generate)
        cost = my_cost(local_cost, server_cost, quad_drift) + float((failed_to_offload+failed_to_generate)>0)*np.exp(failed_to_offload+failed_to_generate+1)

        if not silence: print("cost = {}".format(cost))
        return cost
