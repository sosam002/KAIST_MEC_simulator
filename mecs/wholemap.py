# -*- coding: utf-8 -*-
import csv
import logging
import pathlib
import random

import numpy

from mecs.mobilenode import MobileNode
from mecs.servernode import ServerNode

logger = logging.getLogger(__name__)
_parent = pathlib.Path(__file__).parent


class WholeMap:
    def __init__(self, maxX, maxY, arrival_rate, departure_rate):
        self.unit_time = 0.001  # (second)
        self.minX = 0
        self.minY = 0
        self.maxX = maxX
        self.maxY = maxY
        self.arrival_type = "Poisson"
        self.arrival_rate = arrival_rate
        self.departure_type = "exponential"
        self.departure_rate = departure_rate
        self.mobiles = {}
        self.servers = []
        self.applications = {}
        self.applications_initialize()
        self.index = 0
        self.battery_limit = 250

    def applications_initialize(self):
        with (_parent / 'applications.csv').open(mode='r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                self.applications[row[0]] = (int(row[1]), float(row[2]),
                                             float(row[3]), float(row[4]))
        logger.info(self.applications)

    @staticmethod
    def distance(node_1, node_2):
        return ((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2) ** 0.5

    def add_server(self, x, y, server_capability, schedule_method):
        server = ServerNode(x, y, self, server_capability, schedule_method)
        self.servers.append(server)

    def add_mobile(self, t):
        self.mobiles[self.index] = MobileNode(t, self.index, self)
        self.index += 1

    def mobile_arrive(self, t):
        if random.random() < self.arrival_rate:
            self.add_mobile(t)

    def all_mobiles_move(self, t):
        for _, mobile in self.mobiles.items():
            mobile.move(t)

    def all_mobiles_proceed_and_offload(self, t):
        for _, mobile in self.mobiles.items():
            mobile.transmit_and_proceed(t)

    def all_servers_do_tick(self, t):
        for server in self.servers:
            server.do_tick(t)

    def all_mobiles_log(self, t):
        status = {}
        for i, mobile in self.mobiles.items():
            status['m' + str(i)] = mobile.get_status()
        return status

    def all_servers_log(self, t):
        status = {}
        for server in self.servers:
            status[server.uuid.hex] = server.get_status()
        return status

    def print_all_mobiles(self):
        logger.info("================ Simulation is all over ================")
        logger.info("=========== Printing all mobile nodes' status ===========")
        for mobile in self.mobiles:
            self.mobiles[mobile].print_me()
        logger.info("=========== Printing all server nodes' status ===========")
        for server in self.servers:
            server.print_me()
        logger.info("================ Printing is done ================")

    def mobile_departure(self, t):
        mobiles_departing = []
        for index, mobile in self.mobiles.items():
            r = random.random()
            if (r < self.departure_rate * numpy.log(mobile.living_time + 1)
                or mobile.battery < self.battery_limit) \
                    and mobile.amount_of_data_to_proceed <= 0 \
                    and mobile.amount_of_data_to_offload <= 0:
                logger.info(
                    '[%f]초에 %d번째 모바일이 사라졌습니다: '
                    '[%f] Ws, [%s] types of [%d] bytes',
                    float(t) / 1000, index, mobile.battery,
                    mobile.application_type, mobile.amount_of_data_to_proceed)
                mobiles_departing.append(index)
        self.remove_mobiles(mobiles_departing)
        # TODO : abort the offloaded task

    def remove_mobiles(self, mobiles):
        for index in mobiles:
            del self.mobiles[index]

    def calculate_all_channel_gain(self):
        for _, mobile in self.mobiles.items():
            mobile.calculate_channel_gain()

    def simulate_one_time(self, t):
        self.mobile_arrive(t)
        self.mobile_departure(t)
        self.all_mobiles_move(t)
        self.calculate_all_channel_gain()
        self.all_mobiles_proceed_and_offload(t)
        self.all_servers_do_tick(t)
        log = self.all_servers_log(t)
        log.update(self.all_mobiles_log(t))
        return log
