# import tensorflow as tf
import numpy as np
# 임시로
def local_energy_consumption(used_cpu, used_tx=0):
    return used_cpu**2+used_tx

def offload_cost(used_tx, workloads):
    used_cpu = np.array(used_tx)*np.array(workloads)
    return used_cpu**2

def offload_cost(task_to_be_offloaded):
    used_cpu = 0
    for _, task_ob in task_to_be_offloaded.items():
        used_cpu += task_ob.get_workload()*task_ob.get_data_size()
    return used_cpu**2

def queue_size_penalty(queue_lengths, max_queue_lengths):
    remaind_queues = np.square(max_queue_lengths-queue_lengths)
    return np.sum(remained_queues)

def quad_Lyapunov(queue_list):
    sum = 0
    for queue_id, queue in queue_list.items():
        sum += queue.length**2
    return sum

# 그럴싸한 coefficient 알아와야 함.
def my_rewards(local_cost, server_cost, quad_drift, gamma_1=0.9, gamma_2=0.1, gamma_3=0.5, V=3):
    return V*(gamma_1*local_cost**2*1e-30 + gamma_2 + gamma_3*server_cost**2*1e-30) + quad_drift



class Lyapunov_buffer:
    def __init__(self, max_size=2, initial_storage=0):
        # super.__init__(max_size)
        self.storage = [initial_storage]

        # if initial_storage is None:
        #     self.storage = []
        # else:
        #     self.storage = [initial_storage]
        self.max_size = max_size

    def add(self, data):
        self.storage.append(data)
        if len(self.storage) > self.max_size:
            self.storage = self.storage[1:]
        else:
            pass

    # 참 트루..?
    def get_avg_drift(self):
        return self.storage[-1]-self.storage[0]

    def get_drift(self):
        return self.storage[-1]-self.storage[-2]

    def get_buffer(self):
        return self.storage


class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

# class Buffer(object):
# 	def __init__(self, max_size=100):
# 		self.storage = []
# 		self.max_size = max_size
# 		self.ptr = 0
#
# 	def add(self, data):
# 		if len(self.storage) == self.max_size:
# 			self.storage[int(self.ptr)] = data
# 			self.ptr = (self.ptr + 1) % self.max_size
# 		else:
# 			self.storage.append(data)

#
# class Replay_buffer(Buffer):
#     def __init__(self, max_size=1e6):
#         super.__init__(max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, len(self.storage), size=batch_size)
#         x, y, u, r, d = [], [], [], [], []
#
#         for i in ind:
#             X, Y, U, R, D = self.storage[i]
#             x.append(np.array(X, copy=False))
#             y.append(np.array(Y, copy=False))
#             u.append(np.array(U, copy=False))
#             r.append(np.array(R, copy=False))
#             d.append(np.array(D, copy=False))
#
#         return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
