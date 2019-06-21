import logging
import uuid
import copy
import numpy as np
from node import Node
from task import Task
from task_queue import TaskQueue
from channels import *
import applications
from rl.utilities import *

logger = logging.getLogger(__name__)


class ServerNode(Node):
    def __init__(self, computation_capability):
        super().__init__()
        # self.map = whole_map
        # self.x = x
        # self.y = y
        # 연결된 uuid와 그와 통신하는 protocol과 그 해당하는 rate?이 필요함..protocol에 포함될수도.
        self.links_to_lower = {} # 하위 device와 통신
        # 연결된 uuid와 그와 통신하는 protocol과 그 해당하는 rate?이 필요함..protocol에 포함될수도.
        # self.links_to_higher = {} # 상위 device와 통신
        # 일단은 그냥 BW 써놨음..ㅠㅠ
        self.links_to_higher = {} # 상위 device와 통신
        self.uuid = uuid.uuid4()
        self.mobility = False
        # self.schedule_method = schedule_method
        self.computation_capability = computation_capability  # clocks/tick
        self.number_of_applications = 0
        self.queue_list = {} # 어플마다 각자의 큐가 필요함.
        self.arrival_size_buffer = Lyapunov_buffer(max_size=100, initial_storage=None)

    # default는 오프로드할 상위 링크를 연결하는 것
    def add_link(self, link_to, channel, type=0):
        if type: # 내가 상위링크
            upper = self
            lower = link_to
        else: # 내가 하위링크
            upper = link_to
            lower = self

        lower.links_to_higher[upper.get_uuid()] = {
            'node' : upper,
            'channel' : channel
        }
        upper.links_to_lower[lower.get_uuid()] = {
            'node' : lower,
            'channel' : channel
        }
        return

    def make_application_queues(self, *application_types):
        for application_type in application_types:
            self.queue_list[application_type] = TaskQueue(application_type)
            self.number_of_applications += 1
        return

    # 모든 application에 대한 액션 alpha, 실제 활용한 총 cpu 비율 return
    def do_tasks(self, alpha):
        # app_type_list = applications.app_type_list()
        app_type_list = list(self.queue_list.keys())
        cpu_allocs = dict(zip(app_type_list, alpha))
        for app_type in app_type_list:
            print("### do_task for app_type{} ###".format(app_type))
            if cpu_allocs[app_type] == 0:
                pass
            else:
                my_task_queue = self.queue_list[app_type]
                if my_task_queue.length:
                    cpu_allocs[app_type], _ = my_task_queue.served(cpu_allocs[app_type]*self.computation_capability, type=1)
                else:
                    cpu_allocs[app_type]=0
        # return alpha, served_bits, my_task_queue
        return sum(cpu_allocs.values())


    def _probe(self, bits_to_be_arrived, id_to_offload):
        node_to_offload = self.links_to_higher[id_to_offload]['node']
        # import pdb; pdb.set_trace()
        for app_type, bits in bits_to_be_arrived.items():
            node_app_queue = node_to_offload.queue_list[app_type]
            if node_app_queue.max_length < node_app_queue.length + bits:
                bits_to_be_arrived[app_type] = 0
            else:
                pass
        return bits_to_be_arrived


    '''
    def _amount_of_offload_to(self, policy):
        ids_to_offload, amount_of_offload = policy
        return dict(zip(ids_to_offload, amount_to_offload))
    '''

    def get_channel_rate(self, id_to_offload):
        # import pdb; pdb.set_trace()
        return get_channel_info(self.links_to_higher[id_to_offload]['channel'], 'rate')

    # 모든 application에 대한 액션 alpha, 실제 활용한 총 cpu 비율 return
    def offload_tasks(self, beta, id_to_offload):
        # 문제가 좀 있음.. channel.py에 channel을 좀 손봐야 함
        channel_rate = self.get_channel_rate(id_to_offload)
        app_type_list = list(self.queue_list.keys())
        tx_allocs = dict(zip(app_type_list, np.array(beta)*channel_rate))
        # import pdb; pdb.set_trace()
        tx_allocs = self._probe(tx_allocs, id_to_offload)
        print("## can I offload? tx_allocs bits {} ##".format(tx_allocs))
        # import pdb; pdb.set_trace()
        task_to_be_offloaded = {}
        for app_type in app_type_list:
            if tx_allocs[app_type] ==0 :
                pass
            else:
                my_task_queue = self.queue_list[app_type]
                if my_task_queue.length:
                    tx_allocs[app_type], new_to_be_offloaded = my_task_queue.served(tx_allocs[app_type], type=0)
                    task_to_be_offloaded.update(new_to_be_offloaded)
                else:
                    tx_allocs[app_type]=0
        # return alpha, served_bits, my_task_queue
        return sum(tx_allocs.values()), task_to_be_offloaded

        '''
        task to be offloaded 문제임
        node마다 태스크 큐 만들면서 새로 태스크를 만들어야 하는데, client, server만 알면 되는 게아님
        task id가 넘어가야 하는데 그게 offload 어떻게 연결되지?
        task에 parents id랑 child id남_
        '''

    # _probe에서 받을 수 있는 것만 받기때문에 arrived에서 또 넘치는 걸 체크할 필요 없네.
    def offloaded_tasks(self, tasks, arrival_timestamp):
        for task_id, task_ob in tasks.items():
            task_ob.client_index = task_ob.server_index
            task_ob.server_index = self.get_uuid()
            task_ob.set_arrival_time = arrival_timestamp
            if not self.queue_list[task_ob.application_type].arrived(task_ob):
                print("queue exploded queue exploded i'm an 'offloaded_tasks'")
            # task가 받아지지 않았을 때 role back 해야 하는데 ㅠㅠ
        return

    # 사실 하위 device에서 offload 받은 task를 전해 받아야 함.
    # 일단 simulation을 위해 그냥 만들어 놓음. 하..
    def random_task_generation(self, task_rate, arrival_timestamp, *app_types):
        app_type_list = applications.app_type_list()
        app_type_pop = applications.app_type_pop()
        this_app_type_list = list(self.queue_list.keys())
        random_id = uuid.uuid4()
        # queue_list = {}
        # task_type = np.random.choice(app_type_list, app_type_pop)
        arrival_size = np.zeros(8)
        for app_type, population in app_type_pop:
            if app_type in this_app_type_list:
                data_size = np.random.poisson(task_rate*population)*applications.arrival_bits(app_type)
                if data_size >0:
                    t = Task(app_type, data_size, client_index = random_id.hex, server_index = self.get_uuid(), arrival_timestamp=arrival_timestamp)
                    self.queue_list[app_type].arrived(t)
                    arrival_size[app_type-1]= data_size
                    print("arrival of app_type{} : {}".format(app_type, data_size))
                else:
                    print("no arrival of app_type{} occured".format(app_type))
            else:
                pass
        self.arrival_size_buffer.add(arrival_size)
        return arrival_size


    def print_me(self):
        logger.info('Server %s at (%d,%d)', self.get_uuid(), self.x, self.y)
        for index, task in self.tasks.items():
            if task.is_start:
                logger.debug(
                    'Task %s of %d/%d for %d',
                    index, task.computation_over, task.data_size,
                    task.client_index)
            else:
                logger.debug(
                    'Task %s of %d/%d for %d, not ready',
                    index, task.computation_over, task.data_size,
                    task.client_index)
    #
    def estimate_arrival_rate(self, interval=10):
        buffer = np.array(self.arrival_size_buffer.get_buffer())
        return np.mean(buffer, axis=0)

    def get_status(self):
        # TODO : list to JSON?
        # val = {
        #     'x': self.x, 'y': self.y,
        #     'uuid': self.uuid,
        #     'channels': self.channels,
        #     'node_type': self.node_type,
        #     'tasks': self.tasks,
        #     'computation_capability': self.computation_capability
        # }

        # 앱 개수만큼 리스트
        queue_lengths = np.zeros(8)
        # arrival_rates = np.zeros(8)
        for _, queue in self.queue_list.items():
            queue_lengths[queue.app_type-1] = queue.length
            # arrival_rates[queue.app_type-1] = queue.estimate_arrival_rate()
        # 아 채널 스테이트도 받아와야 하는데 ㅠㅠ 일단 메인에서 받는다

        return list(queue_lengths) + [self.computation_capability]
