import logging
import uuid
import copy
import numpy as np

import applications
from task import Task
from task_queue import TaskQueue

logger = logging.getLogger(__name__)


# class ServerNode(Node):
class ServerNode:
    def __init__(self, computation_capability, is_random_task_generating=False):
        super().__init__()
        # self.map = whole_map
        # self.x = x
        # self.y = y
        # 연결된 uuid와 그와 통신하는 protocol과 그 해당하는 rate?이 필요함..protocol에 포함될수도.
        self.links_to_lower = {} # 하위 device와 통신
        self.links_to_higher = {} # 상위 device와 통신
        self.uuid = uuid.uuid4()
        self.mobility = False
        # self.schedule_method = schedule_method
        self.computation_capability = computation_capability  # clocks/tick
        self.number_of_applications = 0
        self.queue_list = {} # 어플마다 각자의 큐가 필요함.
        self.is_random_task_generating = is_random_task_generating

    def __del__(self):
        iter = list(self.queue_list.keys())
        for app_type in iter:
            del self.queue_list[app_type]
        del self

    def make_application_queues(self, *application_types):
        for application_type in application_types:
            self.queue_list[application_type] = TaskQueue(application_type)
            self.number_of_applications += 1
        return

    # 모든 application에 대한 액션 alpha, 실제 활용한 총 cpu 비율 return
    def do_tasks(self, alpha):
        app_type_list = list(self.queue_list.keys())
        cpu_allocs = dict(zip(app_type_list, alpha))
        for app_type in app_type_list:
            if cpu_allocs[app_type] == 0 or (app_type not in self.queue_list.keys()):
                pass
            else:
                # print("### do_task for app_type{} ###".format(app_type))
                my_task_queue = self.queue_list[app_type]
                if my_task_queue.get_length(None):
                    cpu_allocs[app_type], _ = my_task_queue.served(cpu_allocs[app_type]*self.computation_capability, type=1)
                else:
                    cpu_allocs[app_type]=0

        # return alpha, served_bits, my_task_queue
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
            if self.queue_list[app_type].get_max() < self.queue_list[app_type].get_length(normalize=None)+bits_to_be_arrived:
                return (0, True)
            else:
                return (bits_to_be_arrived, False)
        return (0, True)

    # 모든 application에 대한 액션 beta, id_to_offload로 offload함.  (_probe로 가능한 action으로 바꿔서)
    def offload_tasks(self, beta, id_to_offload):
        channel_rate = self.sample_channel_rate(id_to_offload)
        # app_type_list = applications.app_type_list()
        app_type_list = list(self.queue_list.keys())
        lengths = self.get_queue_lengths()
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
                if my_task_queue.length:
                    tx_allocs[app_type], new_to_be_offloaded = my_task_queue.served(tx_allocs[app_type], type=0)
                    task_to_be_offloaded.update(new_to_be_offloaded)
                else:
                    tx_allocs[app_type]=0
        # return alpha, served_bits, my_task_queue
        return sum(tx_allocs.values()), task_to_be_offloaded, failed

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
    #
    # def random_task_generation(self, task_rate, arrival_timestamp, *app_types):
    #     app_type_list = applications.app_type_list()
    #     app_type_pop = applications.app_type_pop()
    #     this_app_type_list = list(self.queue_list.keys())
    #     arrival_size = np.zeros(len(app_types))
    #     failed_to_generate = 0
    #     for app_type, population in app_type_pop:
    #         if app_type in this_app_type_list:
    #             generated_number = np.random.poisson(task_rate*population)
    #             for i in range(generated_number):
    #                 data_size = applications.arrival_bits(app_type)
    #                 if data_size >0:
    #                     task = Task(app_type, data_size, client_index = uuid.uuid4().hex, server_index = self.get_uuid(), arrival_timestamp=arrival_timestamp)
    #                     failed_to_generate += (not self.queue_list[app_type].arrived(task, arrival_timestamp))
    #                     arrival_size[app_type-1] += data_size
    #                     # print("arrival of app_type{} : {}".format(app_type, data_size))
    #                 else:
    #                     pass
    #                     # print("no arrival of app_type{} occured".format(app_type))
    #         else:
    #             pass
    #     # self.arrival_size_buffer.add(arrival_size)
    #     return arrival_size, failed_to_generate



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

    def get_higher_node_ids(self):
        return list(self.links_to_higher.keys())

    def get_queue_list(self):
        return self.queue_list.items()

    # 이 return값은 딱 이 node queue_list만큼의 길이임.
    def get_queue_lengths(self):
        lengths = np.zeros(len(self.queue_list))
        for app_type, queue in self.queue_list.items():
            lengths[app_type-1]=queue.get_length(normalize=None)
        return lengths

    def sample_channel_rate(self, linked_id):
        if linked_id in self.links_to_higher.keys():
            return self.links_to_higher[linked_id]['channel'].get_rate()
        elif linked_id in self.links_to_lower.keys():
            return self.links_to_lower[linked_id]['channel'].get_rate(False)

    def get_applications(self):
        return list(self.queue_list.keys())

    def get_status(self, time, estimate_interval=100, involve_capability=0):
        queue_estimated_arrivals = np.zeros(8)
        queue_arrivals = np.zeros(8)
        queue_lengths = np.zeros(8)
        queue_exploded = np.zeros(8)
        # arrival_rates = np.zeros(8)
        for app_type, queue in self.queue_list.items():
            queue_estimated_arrivals[app_type-1] = queue.mean_arrival(time, estimate_interval)
            queue_arrivals[app_type-1] = queue.last_arrival(time)
            queue_lengths[app_type-1] = queue.get_length()
            queue_exploded[app_type-1] = queue.is_exploded()
            # if queue.is_exploded()>0:
            #     import pdb; pdb.set_trace()
            # arrival_rates[queue.app_type-1] = queue.estimate_arrival_rate()
        # 아 채널 스테이트도 받아와야 하는데 ㅠㅠ 일단 메인에서 받는다
        if involve_capability:
            return list(queue_lengths) + [self.computation_capability/GHZ]
        return list(queue_estimated_arrivals)+list(queue_arrivals)+list(queue_lengths)+list(queue_exploded)
        # return list(queue_lengths) + [self.computation_capability/1000000000]
