# import uuid
# import logging
# import collections
# import numpy as np
# from baselines import applications
#
# from baselines.task import *
# from baselines.buffers import TaskBuffer
# from baselines.constants import *

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

    def __init__(self, app_type=None, max_length=np.inf):
        self.uuid = uuid.uuid4()
        self.max_length = max_length
        self.tasks = collections.OrderedDict()
        self.app_type = app_type
        self.arrival_size_buffer = TaskBuffer(max_size=100)
        self.exploded = 0
        self.length = 0
        logger.info('Task queue of app. type {} with max length {} is initiallized'.format(app_type, max_length))

    def __del__(self):
        # logger.info('Task queue of app. type {} with max length {} is removed'.format(app_type, max_length))
        if len(self.tasks):
            ids = list(self.tasks.keys())
            for id in ids:
                del self.tasks[id]
        del self.arrival_size_buffer
        del self

    def task_ready(self, task_id):
        self.tasks[task_id].is_start = True
        logger.debug('task %s ready', task_id)

    def remove_task(self, task_id):
        logger.debug('Task %s removed', task_id)
        self.length -= self.tasks[task_id].get_data_size()
        del self.tasks[task_id]


    def remove_multiple_tasks(self, task_list):
        for task_id in task_list:
            logger.debug('Task %s removed', task_id)
            self.length -= self.tasks[task_id].get_data_size()
            del self.tasks[task_id]


    def abort_task(self, task_id):
        logger.info('Task %s aborted', task_id)
        self.length -= self.tasks[task_id].get_data_size()
        self.remove_task(task_id)

    # task 객체를 받음
    # offloaded_tasks에서 받음. random_task_generation에서도 받음.
    def arrived(self, task, arrival_timestamp):
        task_id = task.get_uuid()
        task_length = task.data_size
        self.arrival_size_buffer.add((arrival_timestamp, task_length))
        if self.get_length() + task_length <= self.max_length:
            self.tasks[task_id] = task
            self.length += task_length
            self.exploded = max(0, self.exploded-1)
            return True
        else:
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
            served = 0
            # application queue 이면서 type 일 때
            if (self.app_type and type) :
                # resource unit : cycles --> bits
                to_be_served = int(resource/applications.get_info(self.app_type,'workload'))
            # application queue 가 아니거나 type 이 아닐 때
            else:
                # to_be_served unit: bits
                # if not app_type: resource(to_be_served) unit: cycles
                # if not type: resource(to_be_served) unit: bits
                to_be_served = resource
            if not silence: print("data size to be offloaded : {}".format(to_be_served))
            for task_id, task_ob in self.tasks.items():
                task_size = task_ob.data_size
                # if not app_type: task_size unit: bits --> cycles
                if (type and not self.app_type):
                    task_size *= applications.get_info(task_ob.get_app_type(),"workload")
                if not silence: print("task_size : {}".format(task_size))
                if to_be_served >= task_size:
                    if not silence: print("data size can be served >= task_size case")
                    if not type:
                        offloaded_tasks[task_id] = task_ob
                    task_to_remove.append(task_id)
                    to_be_served -= task_size
                    served += task_size
                    # if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                elif to_be_served > 0:
                    if not silence: print("data size to be offloaded < task_size case")
                    # if this is an application queue
                    if self.app_type:
                        task_size -= to_be_served
                        # offloading
                        if not type:
                            # task_ob data size is adjusted in make_child_task function
                            new_task = task_ob.make_child_task(to_be_served)
                            # print("old_task uuid\t", task_id)
                            # print("new_task uuid\t", new_task.get_uuid())
                            offloaded_tasks[new_task.get_uuid()] = new_task
                        # computation by itself
                        else:
                            self.tasks[task_id].data_size = task_size
                        self.length -= to_be_served
                        self.get_length()
                    # if this is not an application queue
                    # if not app_type: task_size unit: bits --> cycles
                    else:
                        workload = applications.get_info(task_ob.get_app_type(),"workload")
                        # if not app_type: task_size unit: cycles --> bits
                        task_size /= workload
                        # if not app_type: to_be_served unit: cycles --> bits
                        to_be_served = int(to_be_served/workload)
                        task_size -= to_be_served
                        self.tasks[task_id].data_size = task_size
                        self.length -= to_be_served
                        self.get_length()
                        # if not app_type: to_be_served unit: bit --> cycles
                        to_be_served *= workload

                    served += to_be_served
                    # if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                    to_be_served = 0
                else:
                    if not silence and not type : print('All tasks are done in task_queue.served(type=0) - offloaded')
                    if not silence and type : print('All tasks are done in task_queue.served(type=1) - computed')
                    break
            if type and self.app_type:
                resource = served * applications.get_info(self.app_type,'workload')
            else:
                resource = served
            self.remove_multiple_tasks(task_to_remove)
            self.get_length()
            if not silence: print("########### task_queue.served ends ###########")
            return resource, offloaded_tasks

    def mean_arrival(self, t, interval=10, scale=1):
        result = 0
        for time, data_size in self.arrival_size_buffer.get_buffer():
            if time > t - interval:
                result += data_size
            else:
                break
        return result/min(t+1,interval)/scale

    def last_arrival(self, t, scale=1):
        last_data = self.arrival_size_buffer.get_last_obj()
        if last_data:
            time, data_size =last_data
            if time==t:
                return data_size/scale
        return 0

    def get_uuid(self):
        return self.uuid.hex

    def get_length(self, scale=1, cpu_ver=False):
        if not cpu_ver:
            return self.length/scale
        else:
            length = 0
            for task in self.tasks.values():
                length += task.data_size*applications.get_info(task.app_type,'workload')
            return length


    def get_status(self):
        return self.tasks, self.exploded

    def get_tasks(self):
        return self.tasks

    def get_max(self, scale=1):
        return self.max_length/scale

    def is_exploded(self):
        return self.exploded
