import logging
import uuid

from mecs.node import Node

from mecs.task import Task

logger = logging.getLogger(__name__)


class ServerNode(Node):
    def __init__(self, x, y, whole_map, computation_capability,
                 schedule_method):
        super().__init__()
        self.map = whole_map
        self.x = x
        self.y = y
        self.channels = {}
        self.tasks = {}
        self.uuid = uuid.uuid4()
        self.node_type = 1  # 1 : server node
        self.schedule_method = schedule_method
        self.computation_capability = computation_capability  # clocks/tick

    def get_info_about_offload(self, index, application_type,
                               amount_of_offload):
        # task creation
        t = Task(index, self.get_uuid(), application_type, amount_of_offload)
        self.tasks[t.uuid] = t
        logging.debug('task %s of %d for %d created',
                      t.uuid, amount_of_offload, index)

        return t.uuid

    def task_ready(self, index):
        self.tasks[index].is_start = True
        logger.debug('task %s ready', index)

    def remove_task(self, task_id):
        logger.debug('Task %s removed', task_id)
        del self.tasks[task_id]

    def remove_multiple_tasks(self, task_list):
        for index in task_list:
            logger.debug('Task %s removed', index)
            del self.tasks[index]

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

    def get_uuid(self):
        return self.uuid.hex

    def abort_task(self, task_id):
        logger.info('Task %s aborted', task_id)
        self.remove_task(task_id)

    def do_tick(self, t):
        if self.tasks:
            self.tasks = self.schedule_method(self.tasks)
            task_to_remove = []
            for index, task in self.tasks.items():
                if task.is_start:
                    # print("[%d] Server : task %d calculating for %f"
                    # % (t, task.client_index, task.share))
                    task.computation_over += \
                        (task.share * self.computation_capability)
                    logger.debug('[%d] Task %s : %f/%d with share %f',
                                 t, index, task.computation_over,
                                 task.data_size, task.share)

                    if task.computation_over >= task.data_size:
                        logger.debug(
                            '[%d] Server : task of %d - %f data over',
                            t, task.client_index, task.data_size)
                        if task.client_index - 1 in self.map.mobiles:
                            self.map.mobiles[task.client_index - 1] \
                                .return_result(t, task.uuid)
                        task_to_remove.append(task.uuid)
            if task_to_remove:
                self.remove_multiple_tasks(task_to_remove)

            # TODO: @Sangdon or @Sanghong
            # The belows looks unnecessary. Please remove them.
            return
        else:
            return

    def get_status(self):
        # TODO : list to JSON?
        val = {
            'x': self.x, 'y': self.y,
            'uuid': self.uuid,
            'channels': self.channels,
            'node_type': self.node_type,
            'tasks': self.tasks,
            'computation_capability': self.computation_capability
        }

        return val
