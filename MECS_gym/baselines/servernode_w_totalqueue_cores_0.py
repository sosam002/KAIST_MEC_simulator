import logging
import uuid
import copy
import numpy as np
from scipy.stats import uniform
from scipy.stats import levy
from scipy.spatial import distance

import baselines.applications
from baselines.constants import *
from baselines.task import Task
from baselines.task_queue import TaskQueue

logger = logging.getLogger(__name__)


# class ServerNode(Node):
class ServerNode:
    # def __init__(self, num_cores, single_clk, is_random_task_generating=False):
    def __init__(self, num_cores=54, single_clk=4, is_random_task_generating=False, movable=False):
        super().__init__()
        self.uuid = uuid.uuid4()
        self.location = (np.random.rand(2)*1e6).astype(int)
        self.links_to_lower = {} # 하위 device와 통신
        self.links_to_higher = {} # 상위 device와 통신
        self.num_cores = num_cores
        self.single_clk = single_clk*GHZ
        self.computational_capability = self.num_cores*self.single_clk  # clocks/tick
        self.task_queue = TaskQueue()
        self.cpu_used = 0
        self.movable = movable

    def __del__(self):
        del self.task_queue
        del self

    def move(self, type = 'levy'):
        if type =='levy':
            # uniformly distributed angles
            angle = uniform.rvs( size=(n,), loc=.0, scale=2.*np.pi )

            # levy distributed step length
            r = levy.rvs(loc=3, scale=0.5)
            self.location += [r * np.cos(angle), r * np.sin(angle)]
        else:
            pass

    def get_dist(self, node):
        return distance.euclidean(self.location, node.location)

    # 모든 application에 대한 액션 alpha, 실제 활용한 총 cpu 비율 return
    def do_tasks(self):
        if self.task_queue.get_length():
            self.cpu_used, _ = self.task_queue.served(self.computational_capability, type=1)
        else:
            self.cpu_used=0
        return self.cpu_used

    # id_to_offload로 오프로드할 bits_to_be_arrived(data bit list)를 받아서, each app. queue에 받을만한 공간이 있는지 확인
    # {받을 수 있으면 bit그대로, 아니면 0}, {오프로드 실패했는지 bool}
    def probed(self, app_type, bits_to_be_arrived):
        if self.task_queue.get_max() < self.task_queue.get_length()+bits_to_be_arrived:
            return (0, True)
        else:
            return (bits_to_be_arrived, False)

    # _probe에서 받을 수 있는 것만 받기때문에 arrived에서 또 넘치는 걸 체크할 필요 없네.
    def offloaded_tasks(self, tasks, arrival_timestamp):
        failed_to_offload = 0
        for task_id, task_ob in tasks.items():
            task_ob.client_index = task_ob.server_index
            task_ob.server_index = self.get_uuid()
            task_ob.set_arrival_time(arrival_timestamp)
            failed_to_offload += (not self.task_queue.arrived(task_ob, arrival_timestamp))
        return failed_to_offload

    # 사실 하위 device에서 offload 받은 task를 전해 받아야 함.
    def get_higher_node_ids(self):
        return list(self.links_to_higher.keys())

    def get_task_queue_length(self, scale=1):
        return self.task_queue.get_length(scale=scale)

    def sample_channel_rate(self, linked_id):
        if linked_id in self.links_to_higher.keys():
            return self.links_to_higher[linked_id]['channel'].get_rate()
        elif linked_id in self.links_to_lower.keys():
            return self.links_to_lower[linked_id]['channel'].get_rate(False)

    def up_channels_estimate(self):
        dists, rates =  [], []
        for link, link_info in self.links_to_higher.keys():
            dist= self.get_dist( link_info['node'] )
            rate = link_info['channel'].get_rate(dist=dist)
            dists.append(dist/1e5)
            rates.append(rate/GBPS)
        return dists, rates

    def down_channels_estimate(self):
        dists, rates =  [], []
        for link, link_info in self.links_to_lower.keys():
            dist= self.get_dist( link_info['node'] )
            rate = link_info['channel'].get_rate(is_up=False, dist=dist)
            dists.append(dist/1e5)
            rates.append(rate/GBPS)
        return dists, rates

    def get_uuid(self):
        return self.uuid.hex

    def used_cpu_ratio(self):
        return self.cpu_used/self.computational_capability

    def _get_obs(self, time, estimate_interval=100, involve_capability=False, scale=1):
        # print(self.computational_capability)
        # return [self.task_queue.mean_arrival(time, estimate_interval, scale=GHZ),\
        #         self.task_queue.last_arrival(time, scale=GHZ), self.task_queue.get_length(scale),\
        #         self.cpu_used/self.computational_capability]
        return [self.task_queue.mean_arrival(time, estimate_interval, scale=scale),\
                self.task_queue.last_arrival(time, scale=scale), self.task_queue.get_cpu_needs(scale=scale),\
                self.cpu_used/self.computational_capability]
