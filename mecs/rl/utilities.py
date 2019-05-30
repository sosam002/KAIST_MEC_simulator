import tensorflow as tf

# 임시로
def local_energy_consumption(used_cpu, used_tx=0):
    return used_cpu^2+usex_tx

def offload_cost(used_tx, workloads):
    used_cpu = np.array(used_tx)*np.array(workloads)
    return used_cpu^2

def queue_size_penalty(queue_lengths, max_queue_lengths):
    remaind_queues = np.square(max_queue_lengths-queue_lengths)
    return np.sum(remained_queues)
