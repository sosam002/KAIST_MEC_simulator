import uuid
import logging
import collections
import numpy as np
import applications

from task import *
from buffers import TaskBuffer
from constants import *

logger = logging.getLogger(__name__)

class TaskQueue(object):

    def __init__(self, app_type, max_length=10*GB):
        self.uuid = uuid.uuid4()
        self.max_length = max_length
        self.tasks = collections.OrderedDict()
        self.length = 0
        self.app_type = app_type
        self.arrival_size_buffer = TaskBuffer(max_size=100)
        self.exploded = 0
        logger.info('Task queue of app. type {} with max length {} is initiallized'.format(app_type, max_length))

    def __del__(self):
        # logger.info('Task queue of app. type {} with max length {} is removed'.format(app_type, max_length))
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
            self.exploded = min(10, self.exploded+1)
            return False
            # 뭔가 처리를 해줘야함.. arrive 못받았을 때...

    # default(type=1)는 그냥 자기 cpu로 처리하는 것, 0이 offload하는 것
    def served(self, resource, type = 1, silence=True):
        if not silence: print("########### compute or offload : inside of task_queue.served ##########")
        if resource == 0:
            logger.info('No data to be served')
            return
        else:
            task_to_remove = []
            offloaded_tasks = {}
            served_task_bits = 0
            if not type :
                to_be_served = resource
            else:
                to_be_served = int(resource/applications.app_info[self.app_type]['workload'])
            if not silence: print("data size to be offloaded : {}".format(to_be_served))
            for task_id, task_ob in self.tasks.items():
                task_size = task_ob.data_size
                if not silence: print("task_size : {}".format(task_size))
                if to_be_served >= task_size:
                    if not silence: print("data size can be served >= task_size case")
                    if not type:
                        offloaded_tasks[task_id] = task_ob
                    task_to_remove.append(task_id)
                    to_be_served -= task_size
                    self.length -= task_size
                    served_task_bits += task_size
                    if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                elif to_be_served > 0:
                    if not silence: print("data size to be offloaded < task_size case")
                    task_size -= to_be_served
                    if not type:
                        new_task = task_ob.make_child_task(to_be_served)
                        offloaded_tasks[new_task.get_uuid()] = new_task
                    else:
                        self.tasks[task_id].data_size = task_size
                    self.length -= to_be_served
                    served_task_bits += to_be_served
                    if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                    to_be_served = 0
                else:
                    if not silence and not type : print('All tasks are done in task_queue.served(type=0) - offloaded')
                    if not silence and type : print('All tasks are done in task_queue.served(type=1) - computed')
                    break
            if type:
                resource = served_task_bits * applications.app_info[self.app_type]['workload']
            else:
                resource = served_task_bits
            self.remove_multiple_tasks(task_to_remove)
            if not silence: print("########### task_queue.served ends ###########")
            return resource, offloaded_tasks

    def mean_arrival(self, t, interval=10, normalize=100):
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
        last_data = self.arrival_size_buffer.get_last_obj()
        if last_data:
            time, data_size =last_data
            if time==t:
                if not normalize:
                    return data_size
                else:
                    return data_size/self.max_length*normalize
        return 0

    def get_uuid(self):
        return self.uuid.hex

    def get_length(self, normalize=100):
        if not normalize:
            return self.length
        else:
            return self.length/self.max_length*normalize

    def get_status(self):
        return self.tasks, self.exploded

    def get_tasks(self):
        return self.tasks

    def get_max(self):
        return self.max_length

    def is_exploded(self):
        # if self.exploded ==True:
        #     import pdb; pdb.set_trace()
        return self.exploded
