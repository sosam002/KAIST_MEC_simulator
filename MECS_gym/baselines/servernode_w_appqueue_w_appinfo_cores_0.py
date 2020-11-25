import logging
import uuid
import copy
import numpy as np
from scipy.stats import uniform
from scipy.stats import levy
from scipy.spatial import distance

from baselines.constants import *
from baselines import applications
from baselines.task import Task
from baselines.task_queue import TaskQueue

logger = logging.getLogger(__name__)


# class ServerNode(Node):
class ServerNode:
    def __init__(self, num_cores=10, single_clk=4, is_random_task_generating=False, movable = False):
        super().__init__()
        # self.map = whole_map
        self.uuid = uuid.uuid4()
        self.location = (np.random.rand(2)*1e6).astype(int)

        # dict ( id of linked node : { 'node' : obj. of the linked node, 'channel' : channel between this node and the linked node } )
        self.links_to_lower = {} # 하위 device와 통신
        self.links_to_higher = {} # 상위 device와 통신
        self.mobility = False
        # self.schedule_method = schedule_method
        self.num_cores = num_cores
        self.single_clk = single_clk*GHZ
        self.computational_capability = self.num_cores*self.single_clk  # clocks/tick
        self.number_of_applications = 0
        self.queue_list = {} # 어플마다 각자의 큐가 필요함.
        self.is_random_task_generating = is_random_task_generating
        self.cpu_used = {}
        self.temp_tx_allocs = {}
        self.movable = movable

    def __del__(self):
        iter = list(self.queue_list.keys())
        for app_type in iter:
            del self.queue_list[app_type]
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

    def make_application_queues(self, *application_types):
        for application_type in application_types:
            self.queue_list[application_type] = TaskQueue(application_type)
            self.number_of_applications += 1
            self.cpu_used[application_type] = 0
            # self.temp_tx_allocs[application_type] = 0
        self.temp_tx_allocs = dict.fromkeys(self.links_to_higher, dict.fromkeys(self.queue_list, 0))
        return

    # 모든 application에 대한 액션 alpha, 실제 활용한 총 cpu 비율 return
    def do_tasks(self, alpha):
        app_type_list = list(self.queue_list.keys())
        cpu_allocs = dict(zip(app_type_list, alpha))
        # print(app_type_list)
        # print(dict(zip(app_type_list, alpha)))
        for app_type in app_type_list:
            if cpu_allocs[app_type] == 0 or (app_type not in self.queue_list.keys()):
                pass
            else:
                # print("### do_task for app_type{} ###".format(app_type))
                my_task_queue = self.queue_list[app_type]
                if my_task_queue.get_length():
                    cpu_allocs[app_type], _ = my_task_queue.served(cpu_allocs[app_type]*self.computational_capability, type=1)

                else:
                    cpu_allocs[app_type]=0

        # return alpha, served_bits, my_task_queue
        self.cpu_used = cpu_allocs
        return sum(cpu_allocs.values())

    # id_to_offload로 오프로드할 bits_to_be_arrived(data bit list)를 받아서, each app. queue에 받을만한 공간이 있는지 확인
    # {받을 수 있으면 bit그대로, 아니면 0}, {오프로드 실패했는지 bool}
    def _probe(self, bits_to_be_arrived, id_to_offload):
        node_to_offload = self.links_to_higher[id_to_offload]['node']
        failed = {}
        for app_type, bits in bits_to_be_arrived.items():
            if (app_type in self.queue_list.keys()):
                bits_to_be_arrived[app_type], failed[app_type] = node_to_offload.probed(app_type, bits)
        return bits_to_be_arrived, failed

    def probed(self, app_type, bits_to_be_arrived):
        if self.queue_list[app_type]:
            if self.queue_list[app_type].get_max() < self.queue_list[app_type].get_length()+bits_to_be_arrived:
                return (0, True)
            else:
                return (bits_to_be_arrived, False)
        return (0, True)

    # 모든 application에 대한 액션 beta, id_to_offload로 offload함.  (_probe로 가능한 action으로 바꿔서)
    def offload_tasks(self, beta, id_to_offload, offload_type):
        channel_rate = self.sample_channel_rate(id_to_offload)
        # app_type_list = applications.app_type_list()
        app_type_list = list(self.queue_list.keys())
        lengths = self.get_queue_lengths()
        # print(lengths)
        # print(np.array(beta)*channel_rate)
        # tx_allocs = dict(zip(app_type_list, (np.array(beta)*channel_rate).astype(int)))
        # 지금 있는 task보다 더 맡길 수는 없지..
        tx_allocs = dict(zip(app_type_list, np.minimum(lengths,np.array(beta)*channel_rate).astype(int)))
        tx_allocs, failed = self._probe(tx_allocs, id_to_offload)
        # print("## can I offload? tx_allocs bits {} ##".format(tx_allocs))
        task_to_be_offloaded = {}
        for app_type in app_type_list:
            if tx_allocs[app_type] ==0 or (app_type not in self.queue_list.keys()):
                pass
            else:
                my_task_queue = self.queue_list[app_type]
                if my_task_queue.get_length():
                    tx_allocs[app_type], new_to_be_offloaded = my_task_queue.served(tx_allocs[app_type], type=0, offload_type = offload_type)
                    task_to_be_offloaded.update(new_to_be_offloaded)
                else:
                    tx_allocs[app_type]=0
        # return alpha, served_bits, my_task_queue
        self.temp_tx_allocs[id_to_offload].update(tx_allocs)
        return task_to_be_offloaded, failed

    # _probe에서 받을 수 있는 것만 받기때문에 arrived에서 또 넘치는 걸 체크할 필요 없네.
    def offloaded_tasks(self, tasks, arrival_timestamp):
        failed_to_offload = 0
        for task_id, task_ob in tasks.items():
            task_ob.client_index = task_ob.server_index
            task_ob.server_index = self.get_uuid()
            task_ob.set_arrival_time(arrival_timestamp)
            failed_to_offload += (not self.queue_list[task_ob.application_type].arrived(task_ob, arrival_timestamp))
            # self.queue_list[task_ob.application_type].arrived(task_ob, arrival_timestamp)
            # if not self.queue_list[task_ob.application_type].arrived(task_ob):
            #     print("queue exploded queue exploded i'm an 'offloaded_tasks'")
            #     is_exploded.append(True)
            # task가 받아지지 않았을 때 role back 해야 하는데 ㅠㅠ
        return failed_to_offload

    # 사실 하위 device에서 offload 받은 task를 전해 받아야 함.
    # 일단 simulation을 위해 그냥 만들어 놓음. 하..
    def random_task_generation(self, task_rate, arrival_timestamp, *app_types):
        app_type_list = applications.app_type_list()
        app_type_pop = applications.app_type_pop()
        this_app_type_list = list(self.queue_list.keys())
        random_id = uuid.uuid4()
        # queue_list = {}
        # task_type = np.random.choice(app_type_list, app_type_pop)
        arrival_size = np.zeros(len(app_types))
        failed_to_generate = 0
        for app_type, population in app_type_pop:
            if app_type in this_app_type_list:
                data_size = np.random.poisson(task_rate*population)*applications.arrival_bits(app_type)
                if data_size >0:
                    task = Task(app_type, data_size, client_index = random_id.hex, server_index = self.get_uuid(), arrival_timestamp=arrival_timestamp)
                    failed_to_generate += (not self.queue_list[app_type].arrived(task, arrival_timestamp))
                    arrival_size[app_type-1]= data_size
                    # print("arrival of app_type{} : {}".format(app_type, data_size))
                else:
                    pass
                    # print("no arrival of app_type{} occured".format(app_type))
            else:
                pass
        # self.arrival_size_buffer.add(arrival_size)
        return arrival_size, failed_to_generate

    def get_higher_node_ids(self):
        return list(self.links_to_higher.keys())

    def get_queue_list(self):
        return self.queue_list.items()

    # 이 return값은 딱 이 node queue_list만큼의 길이임.
    def get_queue_lengths(self, scale = 1):
        lengths = np.zeros(len(self.queue_list))
        for app_type, queue in self.queue_list.items():
            lengths[app_type-1]=queue.get_length(scale)
        return lengths

    def sample_channel_rate(self, linked_id):
        if linked_id in self.links_to_higher.keys():
            return self.links_to_higher[linked_id]['channel'].get_rate()
        elif linked_id in self.links_to_lower.keys():
            return self.links_to_lower[linked_id]['channel'].get_rate(False)

    def get_applications(self):
        return list(self.queue_list.keys())

    def get_uuid(self):
        return self.uuid.hex

    def get_req_cpus_offloaded(self, scale=1):
        req_cpus_offloaded = {}
        for link, temp_tx_alloc  in self.temp_tx_allocs.items():
            req_cpu_offloaded =0
            for app_type, queue in self.queue_list.items():
                req_cpu_offloaded += temp_tx_alloc[app_type]*applications.get_info(app_type, "workload")/scale
            req_cpus_offloaded[link]=req_cpu_offloaded
        return req_cpus_offloaded

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

    def _get_obs(self, time, estimate_interval=100, scale=1, involve_capability=False, involve_dist_channel=False):
        queue_estimated_arrivals = np.zeros(self.number_of_applications)
        queue_arrivals = np.zeros(self.number_of_applications)
        queue_lengths = np.zeros(self.number_of_applications)
        app_info = np.zeros(self.number_of_applications)
        cpu_used = np.zeros(self.number_of_applications)
        # arrival_rates = np.zeros(3)
        for app_type, queue in self.queue_list.items():
            queue_estimated_arrivals[app_type-1] = queue.mean_arrival(time, estimate_interval, scale=scale)
            queue_arrivals[app_type-1] = queue.last_arrival(time, scale=scale)
            queue_lengths[app_type-1] = queue.get_length(scale=scale)
            app_info[app_type-1] = applications.get_info(app_type, "workload")/KB
            cpu_used[app_type-1] = self.cpu_used[app_type]/self.computational_capability

        # print("asdsad")
        # print(self.cpu_used)
        # print(self.computational_capability)
        # print(queue_lengths)
        # print(cpu_used)
            # if queue.is_exploded()>0:
            #     import pdb; pdb.set_trace()
            # arrival_rates[queue.app_type-1] = queue.estimate_arrival_rate()
        # 아 채널 스테이트도 받아와야 하는데 ㅠㅠ 일단 메인에서 받는다
        obs = list(queue_estimated_arrivals)+list(queue_arrivals)+list(queue_lengths)+list(cpu_used)+list(app_info)+list(self.get_req_cpus_offloaded(scale).values())
        if involve_capability:
            obs+=[self.computational_capability/GHZ]
        if involve_dist_channel:
            u_dists, u_rates = self.up_channels_estimate()
            d_dists, d_rates = self.down_channels_estimate()
            obs+=u_dists+u_rates+d_dists+d_rates
        # return list(queue_estimated_arrivals)+list(queue_arrivals)+list(queue_lengths)+list(cpu_used)+list(app_info)\
        #     +list(self.get_req_cpus_offloaded(scale).values()) + self.up_channels_estimate() + self.down_channels_estimate()
        # return list(queue_lengths) + [self.computational_capability/1000000000]
        return obs
