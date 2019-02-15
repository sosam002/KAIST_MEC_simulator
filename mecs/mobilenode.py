# -*- coding: utf-8 -*-
import logging
import random

import numpy

from mecs import mobility
from mecs.node import Node

logger = logging.getLogger(__name__)

# Watt second 기준: Ws = Wh * 3600, LG G5 배터리 기준 10.8 Wh
battery_max = 10.8 * 3600
unit_time = 0.001  # (second)


class MobileNode(Node):

    def __init__(self, t, index, whole_map):
        super().__init__()
        self.map = whole_map
        self.walk = mobility.truncated_levy_walk(
            1, dimensions=(self.map.maxX, self.map.maxY))
        self.x = next(self.walk)[0][0]
        self.y = next(self.walk)[0][1]
        self.index = index + 1
        self.living_time = 0  # [millisecond]
        self.battery = random.randrange(battery_max / 2, battery_max)
        # [cycles/millisecond]
        self.number_of_cpu_cycles_per_millisecond = 2 * 10 ** 9 * unit_time

        # 삼성 갤럭시 노트8 기준:
        #   2 GHz CPU = 2*10^9 cycles/second = 2*10^6 cycles/millisecond
        self.transmission_power_to_offload = 3  # [W]

        # [Power Consumption Analysis of a Modern Smartphone] 논문 기준으로,
        # LTE 전송을 계속 했을 때 3.6시간 정도 사용 가능했으므로,
        # 10.8 Wh 를 3.6시간동안 사용하다면 대략 3 W 의 파워로
        # 전송하는 셈이므로, 우선 3으로 잡았음.
        self.amount_of_data_to_proceed = 0  # [bytes]
        self.application_type = ''
        self.amount_of_data_to_offload = 0  # [bytes]
        self.money_to_pay = 0
        self.happiness = 0
        self.bandwidth = 5 * 10 ** 6  # [Hz]

        # LTE-A 기준으로 일반적으로 대역폭은 1.5 MHz, 3 MHz, 5 MHz, 10 MHz,
        # 15 MHz, 20 MHz 중에 할당됨
        # 본 시뮬레이터에서는 우선 5 MHz를 default 로 사용하고 있음
        self.server = self.select_server()
        self.channel_gain = self.calculate_channel_gain()
        self.energy_for_one_cpu_cycle = 2 * 10 ** (-9)  # ( Ws / cycle )

        self.node_type = 2  # 2 : Mobile node

        self.task_index = ''
        self.total_data = 0
        # TODO : separation of AP and server

    def determine_how_much_to_offload(self):
        self.amount_of_data_to_offload = self.amount_of_data_to_proceed // 2
        self.amount_of_data_to_proceed -= self.amount_of_data_to_offload

        logger.info(
            '오프로딩할 데이터의 양은 [%d]-bytes, 직접 처리할 데이터의 양은 '
            '[%d]-bytes로 결정하였습니다.',
            self.amount_of_data_to_offload, self.amount_of_data_to_proceed)
        self.tell_to_server_how_much_to_offload()

    def tell_to_server_how_much_to_offload(self):
        """ request offloading """
        self.task_index = self.server.get_info_about_offload(
            self.index, self.application_type, self.amount_of_data_to_offload)

    def transmit_and_proceed(self, t):
        random_to_start = random.random()
        if self.amount_of_data_to_proceed == 0 \
                and self.amount_of_data_to_offload == 0:
            # 프로세싱할 무거운 작업이 없을 때
            # (웹서핑 등 간단한 작업만 하고 있음)

            if random_to_start < 0.0001:  # 일정 확률로 헤비한 작업을 시작한다.
                # 어떠한 무거운 작업을 시작할 것인지 선택하러 함수로 들어가자.
                self.determine_which_job_to_start(t)
                # 이제 얼마나 오프로딩을 할 지 최적의 결정을 하러 함수로 들어가자.
                self.determine_how_much_to_offload()
            # (가정에 따라 지금 한번만) 오프로딩을 위한 데이터를 전송하는
            # 최적의 파워를 결정하러 가자.
            # self.determine_how_much_trans_power_for_offloading()
        else:
            # 프로세싱할 작업도 있고,
            # 오프로딩 할 데이터의 양도 결정이 되어 있으면
            if self.living_time % 1000 == 0:
                # 1초마다 한번씩은 (주기는 바뀔 수 있음)

                # 오프로딩을 위한 데이터를 전송하는 최적의 파워를 결정하러 가자.
                self.determine_how_much_trans_power_for_offloading()

                # 내가 직접 수행할 파워도 최적 결정을 해 볼 수도 있으니,
                # 일단 적어놓음.
                # self.determine_how_much_power_to_proceed_heavy_data()

            # if self.living_time % 1000 == 0:
            #     logger.debug('%d번째 모바일의 헤비작업 전의 배터리 양: '
            #                  '[%d] mWs', self.index, self.battery)

            # TODO: @Sangdon
            # If amount data < 0, there is no data to be transmit.
            # So it would be better to check amount > 0 instead amount != 0.
            if self.amount_of_data_to_offload != 0:
                # 오프로딩 할 데이터가 남아있으면
                # 오프로딩 할 데이터를 전송하자.
                self.transmit_offloading_data(t)
            if self.amount_of_data_to_proceed != 0:
                # 헤비한 작업이 남아 있으면
                # 내가 직접 헤비한 작업도 하긴 해야하니, 그 작업도 수행하자.
                self.proceed_heavy_data_myself(t)
            # if self.living_time % 1000 == 0:
            #     logger.debug('%d번째 모바일의 헤비작업 후의 배터리 양: '
            #                  '[%d] mWs', self.index, self.battery)

    def determine_how_much_trans_power_for_offloading(self):
        """
        1초마다 한번씩 오프로딩을 위한 데이터를 전송하는 최적의 파워를 결정
        """
        self.transmission_power_to_offload = 3

    def determine_which_job_to_start(self, t):
        random_for_selecting_application_type = random.random()
        popularity_sum = 0
        for app_type in self.map.applications:
            popularity_sum += self.map.applications[app_type][1]
            if random_for_selecting_application_type < popularity_sum:
                self.application_type = app_type
                min_byte = self.map.applications[app_type][2]
                max_byte = self.map.applications[app_type][3]
                self.amount_of_data_to_proceed = random.randrange(min_byte,
                                                                  max_byte)
                self.total_data = self.amount_of_data_to_proceed

                logger.info(
                    '[%f]초에 %d번째 모바일이 [%s] type의 [%d] bytes에 '
                    '해당하는 작업을 시작하였습니다. 배터리 잔량: [%d mWs]',
                    float(t) / 1000, self.index, self.application_type,
                    self.amount_of_data_to_proceed, self.battery)
                return

    def transmit_offloading_data(self, t):
        self.battery -= self.transmission_power_to_offload * unit_time
        maximum_data_rate = unit_time * self.bandwidth * numpy.log(
            1 + self.transmission_power_to_offload * (
                    self.channel_gain ** 2) / random.randrange(1, 2))
        self.amount_of_data_to_offload -= min(self.amount_of_data_to_offload,
                                              maximum_data_rate)
        if self.amount_of_data_to_offload == 0:
            self.server.task_ready(self.task_index)
            logger.info(
                '[%f]초에 %d번째 모바일의 [%s] 타입의 오프로딩 데이터의 '
                '전송 작업이 끝났습니다. 배터리 잔량: [%d mWs] ',
                float(t) / 1000, self.index, self.application_type,
                self.battery)

    def proceed_heavy_data_myself(self, t):
        cpu_cycles_for_one_byte_process = \
            self.map.applications[self.application_type][0]
        bytes_per_millisecond = self.number_of_cpu_cycles_per_millisecond \
            / cpu_cycles_for_one_byte_process
        self.amount_of_data_to_proceed -= min(self.amount_of_data_to_proceed,
                                              bytes_per_millisecond)
        self.battery -= self.energy_for_one_cpu_cycle \
            * cpu_cycles_for_one_byte_process * bytes_per_millisecond
        if self.amount_of_data_to_proceed == 0:
            logger.info(
                '[%f]초에 %d번째 모바일의 [%s] 타입의 헤비 프로세스가 '
                '끝났습니다. 배터리 잔량: [%d mWs]',
                float(t) / 1000, self.index, self.application_type,
                self.battery)

    def calculate_channel_gain(self):
        dist = self.map.distance(self.server, self)
        return random.randrange(3000, 4000) / dist ** 2

    def return_result(self, t, task_id):
        logger.info('[%d] Mobile %d - Offloaded Task %s is over',
                    t, self.index, task_id)

    def select_server(self):
        for server in self.map.servers:
            selected_server = server
            return selected_server

    def move(self, t):
        prev_x = self.x
        prev_y = self.y
        walk_now = next(self.walk)
        self.x = walk_now[0][0]
        self.y = walk_now[0][1]
        if self.living_time == 0:
            logger.info(
                '[%f]초에 %d번째 새로운 모바일이 (%f,%f)에서 [%d mWs]의 '
                '에너지를 가지고 생성되었습니다.',
                float(t) / 1000, self.index, self.x, self.y,
                self.battery)
        # else:
        #     logger.debug('(Move) The %dth mobile node at (%f,%f) '
        #                  'is moving to (%f,%f)',
        #                  self.index, prev_x, prev_y, self.x, self.y)
        self.living_time += 1

    def print_me(self):
        logger.info(
            '%d번째 모바일이 (%f,%f)에 있으며, [%d/%d]-bytes의 [%s] 타입 '
            '데이터를 처리중입니다. 배터리 잔량: [%d mWs], Task ID %s',
            self.index, self.x, self.y, self.amount_of_data_to_proceed,
            self.total_data, self.application_type, self.battery,
            self.task_index)

    def get_status(self):
        val = {
            'x': self.x, 'y': self.y,
            # TODO : check next waypoint
            'living_time': self.living_time,
            'battery': self.battery,
            'number_of_cpu_cycles_per_millisecond':
                self.number_of_cpu_cycles_per_millisecond,
            'transmission_power_to_offload':
                self.transmission_power_to_offload,
            'amount_of_data_to_proceed': self.amount_of_data_to_proceed,
            'application_type': self.application_type,
            'amount_of_data_to_offload': self.amount_of_data_to_offload,
            'money_to_pay': self.money_to_pay,
            'happiness': self.happiness,
            'bandwidth': self.bandwidth,
            'node_type': self.node_type,
            'target_server': self.server.get_uuid(),
            'channel_gain': self.channel_gain,
            'energy_for_one_cpu_cycle': self.energy_for_one_cpu_cycle
        }

        return val
