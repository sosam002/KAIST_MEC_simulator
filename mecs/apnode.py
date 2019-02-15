import logging
import uuid

from mecs.node import Node

logger = logging.getLogger(__name__)


class APNode(Node):
    def __init__(self, x, y, isWorkerEnabled):
        super().__init__()
        self.x = x
        self.y = y
        self.channels = {}
        self.tasks = []
        self.uuid = uuid.uuid4()
        self.node_type = 3  # 3 : AP node

    def get_info_about_offload(self, index, application_type,
                               amount_of_offload):
        return

    def print_me(self):
        logger.info('Server at (%d,%d)' % (self.x, self.y))

    def get_uuid(self):
        return self.uuid.hex

    def get_status(self):
        return {'x': self.x, 'y': self.y,
                'channels': self.channels,
                'node_type': self.node_type,
                'tasks': self.tasks}
