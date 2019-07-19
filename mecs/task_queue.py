import uuid
import logging
from abc import abstractmethod, ABCMeta
import collections
import copy
import numpy as np
import applications
# from mecs.task import Task
from task import *
from utilities import *
from constants import *

logger = logging.getLogger(__name__)

class TaskQueue(object):

    def __init__(self, app_type, max_length=100000000*BYTE):
        self.uuid = uuid.uuid4()
        self.max_length = max_length
        self.tasks = collections.OrderedDict()
        self.length = 0
        self.app_type = app_type
        self.arrival_size_buffer = Lyapunov_buffer(max_size=100)
        self.exploded = 0

    def __del__(self):
        if len(self.tasks):
            ids = list(self.tasks.keys())
            for id in ids:
                del self.tasks[id]
        self.length=0
        del self.arrival_size_buffer
        del self

    def task_ready(self, task_id):
        self.tasks[task_id].is_start = True
        logger.debug('task %s ready', task_id)

    def remove_task(self, task_id):
        logger.debug('Task %s removed', task_id)
        del self.tasks[task_id]

    def remove_multiple_tasks(self, task_list):
        for task_id in task_list:
            logger.debug('Task %s removed', task_id)
            del self.tasks[task_id]

    def abort_task(self, task_id):
        logger.info('Task %s aborted', task_id)
        self.remove_task(task_id)

    # task 객체를 받음
    # offloaded_tasks에서 받음. random_task_generation에서도 받음.
    def arrived(self, task, arrival_timestamp):
        task_id = task.get_uuid()
        task_length = task.data_size
        self.arrival_size_buffer.add((arrival_timestamp, task_length))
        new_length = self.length + task_length
        if new_length <= self.max_length:
            self.tasks[task_id] = task
            self.length = new_length
            # logger.info('task arrival success, queuelength {}'.format(self.length))
            self.exploded = max(0, self.exploded-1)
            return True
        else:
            logger.info('queue exploded, app type {}, queuelength {}'.format(self.app_type, self.length))
            del task
            self.exploded += 1
            return False
            # 뭔가 처리를 해줘야함.. arrive 못받았을 때...

    # default(type=1)는 그냥 자기 cpu로 처리하는 것
    def served(self, resource, type = 1, silence=True):

        if not silence: print("########### compute or offload : inside of task_queue.served ##########")
        if resource == 0:
            return
        else:
            task_to_remove = []
            offloaded_tasks = {}
            if type:
                if np.isnan(resource/applications.app_info[self.app_type]['workload']):
                    import pdb; pdb.set_trace()
                to_be_served = int(resource/applications.app_info[self.app_type]['workload'])
                served_task_bits = 0
                if not silence: print("data size to be served : {}".format(to_be_served))
                for task_id, task_ob in self.tasks.items():
                    task_size = task_ob.data_size
                    if not silence: print("task_size : {}".format(task_size))
                    if to_be_served >= task_size:
                        if not silence: print("data size can be served >= task_size case")
                        task_to_remove.append(task_id)
                        to_be_served = to_be_served - task_size
                        served_task_bits += task_size
                        self.length = self.length - task_size
                        if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                    elif to_be_served > 0:
                        if not silence: print("data size to be served < task_size case")
                        task_size = task_size - to_be_served
                        self.tasks[task_id].data_size = task_size
                        self.length = self.length - to_be_served
                        served_task_bits += to_be_served
                        if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                        to_be_served = 0
                    else:
                        if not silence: print("no more data to be served remained! {}".format(to_be_served))
                        if not silence: print('All tasks are done in task_queue.served(type=1) - computed')
                        break
                used_resource = served_task_bits*applications.app_info[self.app_type]['workload']
            else:
                to_be_offloaded = resource
                if not silence: print("data size to be offloaded : {}".format(to_be_offloaded))
                for task_id, task_ob in self.tasks.items():
                    task_size = task_ob.data_size
                    if not silence: print("task_size : {}".format(task_size))
                    if to_be_offloaded >= task_size:
                        if not silence: print("data size can be offloaded >= task_size case")
                        offloaded_tasks[task_id] = task_ob
                        task_to_remove.append(task_id)
                        to_be_offloaded = to_be_offloaded - task_size
                        self.length = self.length - task_size
                        if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                    elif to_be_offloaded > 0:
                        if not silence: print("data size to be offloaded < task_size case")
                        new_task = task_ob.make_child_task(to_be_offloaded)
                        offloaded_tasks[new_task.get_uuid()] = new_task
                        task_size = task_size - to_be_offloaded
                        self.length = self.length - to_be_offloaded
                        if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                        to_be_offloaded = 0
                    else:
                        if not silence: print('All tasks are done in task_queue.served(type=0) - offloaded')
                        break
                used_resource = resource - to_be_offloaded
            self.remove_multiple_tasks(task_to_remove)
            if not silence: print("########### task_queue.served ends ###########")
            return used_resource, offloaded_tasks

    def mean_arrival(self, t, interval=10, normalize=100):
        # import pdb; pdb.set_trace()
        result = 0
        for time, data_size in self.arrival_size_buffer.get_buffer():
            if time > t - interval:
                result += data_size
            else:
                break
        if not normalize:
            return result/min(t+1,interval)
        else:
            return result/min(t+1,interval)/self.max_length*normalize

    def last_arrival(self, t, normalize=100):
        time, arrival_size = self.arrival_size_buffer.last_storage()
        if time==t:
            if not normalize:
                return arrival_size
            else:
                return arrival_size/self.max_length*normalize
        else:
            return 0


    @abstractmethod
    def print_me(self):
        pass

    def get_uuid(self):
        return self.uuid.hex

    def get_length(self, normalize=100):
        if not normalize:
            return self.length
        else:
            return self.length/self.max_length*normalize

    def get_status(self):
        return self.tasks, self.exploded

    def get_max(self):
        return self.max_length

    def is_exploded(self):
        # if self.exploded ==True:
        #     import pdb; pdb.set_trace()
        return self.exploded
