import uuid
from abc import abstractmethod, ABCMeta


class Environment(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self.state_dim= 0
        self.action_dim= 0
        self.clients = dict()
        self.servers = dict()
        self.links = list()
        self.timestamp = 0
        self.silence = True

    @abstractmethod
    # reset the environment
    def reset(self):
        pass

    @abstractmethod
    # get the present observation of this environment
    def get_status(self):
        pass

    @abstractmethod
    # do an action and get an observation
    def step(self):
        pass

    @abstractmethod
    # cost function
    def get_cost(self):
        pass

    def silence_on(self):
        self.silence = True
    def silence_off(self):
        self.silence = False
