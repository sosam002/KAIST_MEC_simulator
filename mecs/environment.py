# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy

# from mecs.mobilenode import MobileNode
# from mecs.servernode import ServerNode
from servernode_w_queue import ServerNode

from applications import *
from channels import *
from rl.utilities import *
from constants import *

logger = logging.getLogger(__name__)
_parent = pathlib.Path(__file__).parent


class Environment_sosam:
    def __init__(self, task_rate, *applications, time_delta=10*MS):
        self.task_rate = 10/time_delta
        self.clients = []
        self.servers = []
        self.links = dict()
        self.quad_Lyapunov_buffer = Lyapunov_buffer()
        self.applications = applications
        self.reset_infos = []


    def reset(self):
        task_rate = self.task_rate
        applications = self.applications
        reset_infos = self.reset_infos
        self.__init__(task_rate, *applications)
        for reset_info in reset_infos:
            self.init_for_sosam(*reset_info)
        return self.get_status()

    def add_client(self, cap):
        self.clients.append(ServerNode(cap))
        return

    def add_server(self, cap):
        self.servers.append(ServerNode(cap))
        return

    def add_link(self, client_num, server_num, up_channel, down_channel=None):
        if not down_channel:
            down_channel = up_channel
        # import pdb; pdb.set_trace()
        client = self.clients[client_num]
        server = self.servers[server_num]
        client.links_to_higher[server.get_uuid()]= {
            'node' : server,
            'channel' : up_channel
        }
        server.links_to_lower[client.get_uuid()] = {
            'node' : client,
            'channel' : down_channel
        }
        self.links[(client_num, server_num)] = [down_channel, up_channel]
        return


    def init_for_sosam(self, edge_capability, cloud_capability, channel):
        # 짜증...ㅜㅜ

        self.clients.append(ServerNode(edge_capability))
        self.servers.append(ServerNode(cloud_capability))
        ##
        self.add_link(0, 0, channel)

        self.clients[0].make_application_queues(*self.applications)
        self.servers[0].make_application_queues(*self.applications)
        self.reset_infos.append((edge_capability, cloud_capability, channel))
        return self.get_status()

    def action_space(self):
        pass

    def print_queue_lists(self):
        for node in self.clients:
            for app_type, queue_list in node.queue_list.items():
                print("client server {} tasks {}".format(app_type, queue_list.tasks))
        for node in self.servers:
            for app_type, queue_list in node.queue_list.items():
                print("cloud server {} tasks {}".format(app_type, queue_list.tasks))

        return

    def step(self, action_alpha, action_beta, action_cloud, time, generate_random_task=True, silence =True):
        ######################################################################################## 이게 모조리 step 함수
        # perform action (simulation)
        if generate_random_task:
            if not silence: print("###### random task generation start! ######")
            arrival_size = self.clients[0].random_task_generation(self.task_rate, time, *self.applications)
            if not silence: print("###### random task generation ends! ######")
            # 이건 진짜 arrival rate 이 아님.. arrival만 저장하는걸 또 따로 만들어야 한다니 고통스럽다.
            # print("random task arrival size {}".format(arrival_size))
        #
        # print("edge server AR tasks {}".format(self.clients[0].queue_list[AR].tasks))
        # print("edge server VR tasks {}".format(self.clients[0].queue_list[VR].tasks))
        # print("cloud server AR tasks {}".format(self.servers[0].queue_list[AR].tasks))
        # print("cloud server VR tasks {}".format(self.servers[0].queue_list[VR].tasks))
        # self.print_queue_lists()

        used_edge_cpu = self.clients[0].do_tasks(action_alpha)
        # print("do task on edge, CPU used {}".format(used_edge_cpu))
        used_tx, task_to_be_offloaded = self.clients[0].offload_tasks(action_beta, self.servers[0].get_uuid())
        # print("offload task to cloud, used_tx {}, offloaded task {}".format(used_tx, task_to_be_offloaded))
        self.servers[0].offloaded_tasks(task_to_be_offloaded, time)
        used_cloud_cpu = self.servers[0].do_tasks(action_cloud)
        # print("do task on cloud, CPU used {}".format(used_cloud_cpu))

    ######################################################################################## 이게 모조리 step 함수

        state = self.get_status()
        cost = self.get_reward(used_edge_cpu, task_to_be_offloaded)

        return state, -cost

    def get_status(self):
        estimated_arrival_rate = list(self.clients[0].estimate_arrival_rate())
        # print("estimated_arrival_rate_state = {}".format(estimated_arrival_rate))
        edge_state = self.clients[0].get_status()
        # print("edge state (queue length+cpu cap.) = {}".format(edge_state))
        cloud_state = self.servers[0].get_status()
        # print("cloud state (queue length+cpu cap.) = {}".format(cloud_state))
        # state = estimated_arrival_rate + edge_state + cloud_state + [self.clients[0].get_channel_rate(self.servers[0].get_uuid())/1000000000]
        state = estimated_arrival_rate + edge_state + cloud_state + [self.clients[0].get_channel_rate(self.servers[0].get_uuid())]
        # print("states + channel rate = {}".format(state))
        state = np.log(np.array(state).clip(1))
        return state

    def get_reward(self, used_edge_cpu, task_to_be_offloaded, silence = True):
        self.quad_Lyapunov_buffer.add(quad_Lyapunov(self.clients[0].queue_list)+quad_Lyapunov(self.servers[0].queue_list))
        quad_drift = self.quad_Lyapunov_buffer.get_drift()
        local_cost = local_energy_consumption(used_edge_cpu)
        server_cost = offload_cost(task_to_be_offloaded)
        cost = my_rewards(local_cost, server_cost, quad_drift)*1e-20
        if not silence: print("cost = {}".format(cost))
        return cost
