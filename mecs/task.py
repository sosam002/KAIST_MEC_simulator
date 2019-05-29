import uuid
import copy

class Task(object):
    def __init__(self, client_index, server_index, application_type, data_size):
        self.client_index = client_index
        self.server_index = server_index
        self.application_type = application_type
        self.data_size = data_size
        self.is_start = False
        self.computation_over = 0
        self.uuid = uuid.uuid4()
        self.received_data_size = 0
        self.parent_uuid = None
        self.child_uuid = None

    def get_uuid(self):
        return self.uuid.hex

    def make_child_task(self, offload_data_bits):
        new_task = copy.deepcopy(self)
        new_task.parent_uuid = self.get_uuid()
        self.child_uuid = new_task.get_uuid()
        new_task.data_size = offload_data_bits
        self.data_size = self.data_size-offload_data_bits
        return new_task
