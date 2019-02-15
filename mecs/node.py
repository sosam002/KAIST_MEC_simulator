import uuid
from abc import abstractmethod, ABCMeta


class Node(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self.uuid = uuid.uuid4()
        self.node_type = 0

    @abstractmethod
    def print_me(self):
        pass

    def get_uuid(self):
        return self.uuid.hex

    @abstractmethod
    def get_status(self):
        # TODO: @Sangdon
        # An abstract method should not be implemented.
        # Please remove this implementation or @abstractmethod
        val = dict(x=self.x, y=self.y, node_type=self.node_type)
        return val
