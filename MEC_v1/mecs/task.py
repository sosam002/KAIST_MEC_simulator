import uuid
import copy
import applications

class Task(object):
    def __init__(self, application_type, data_size, client_index = None, server_index = None, arrival_timestamp=None):
        self.client_index = client_index # 이게 오프로드된 작업이라면 요청한 client가 있지
        self.server_index = server_index # 이게 여기서 오프로드시켰다면 오프로드 시킨 server가 있지.
        self.application_type = application_type
        self.data_size = data_size
        self.is_start = False
        self.computation_over = 0
        self.uuid = uuid.uuid4()
        self.received_data_size = 0
        self.parent_uuid = None
        self.child_uuid = None
        # arrival rate 분석할 때 최근 몇 초 안에 도착한 arrival을 기록해야 함.
        self.arrival_timestamp = arrival_timestamp
        self.start_timestamp = None # 혹시 몰라서. task별 waiting time 필요할 수도 있음.
        self.end_timestamp = None # 마찬가지.

    # def __del__(self):
    #     print("deleted")

    def get_uuid(self):
        return self.uuid.hex

    def make_child_task(self, offload_data_bits):
        new_task = copy.deepcopy(self)
        new_task.parent_uuid = self.get_uuid()
        self.child_uuid = new_task.get_uuid()
        new_task.data_size = offload_data_bits
        self.data_size = self.data_size-offload_data_bits
        return new_task

    def get_workload(self):
        return applications.app_info[self.application_type]['workload']

    def get_application_type(self):
        return self.application_type

    def get_data_size(self):
        return self.data_size

    def set_arrival_time(self, arrival_timestamp):
        self.arrival_timestamp = arrival_timestamp

    def get_arrival_time(self):
        return self.arrival_timestamp

    def is_client_index(self):
        return bool(self.client_index)

    def is_server_index(self):
        return bool(self.server_index)

    '''
    self.client_index self.server_index
    0 None , None : 작업 끝나면 끝. 원래 내거
    1 None , is : 작업 끝나면, offload된 거 기다려야 함.
    2 is , None : 작업 끝나면, downlink로 보내야 함. (downlink queue로 가야 함)
    3 is , is : 작업 끝나면, 어딘가에서 output data를 들고 기다리다가... uplink에서 오는거 기다렸다가 downlink로 보내야 함.(downlink queue로 가야 함)
    '''
    def offload_state(self):
        if self.is_client_index() and self.is_server_index():
            return 3
        elif self.is_client_index() and (not self.is_server_index()):
            return 2
        elif (not self.is_client_index()) and self.is_server_index():
            return 1
        else :
            return 0
