from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def schedule(self, task_list):
        raise NotImplementedError()


class RRScheduler(Scheduler):

    def schedule(self, task_list):
        n = len([t for k, t in task_list.items() if t.is_start])
        if n > 0:
            for k, task in task_list.items():
                task.share = 1.0 / n
        return task_list
