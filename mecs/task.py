import uuid


class Task(object):
    def __init__(self, client_index, server_index, application_type, data_size):
        self.client_index = client_index
        self.server_index = server_index
        self.application_type = application_type
        self.data_size = data_size
        self.is_start = False
        self.computation_over = 0
        self.uuid = uuid.uuid4().hex
        self.received_data_size = 0
