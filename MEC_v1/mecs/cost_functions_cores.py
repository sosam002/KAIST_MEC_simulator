import numpy as np
from constants import *

# naive power consumption at a local CPU
# def get_local_power_cost(used_cpu, cores, used_tx=0):
#     return cores*(used_cpu/cores)**3

def get_local_power_cost(used_cpu, used_tx=0):
    cores, remained = divmod(used_cpu, 4*GHZ)
    return cores*(4*GHZ)**3+(remained)**3

def get_fail_cost(failed_to_offload, failed_to_generate):
     return float((failed_to_offload+failed_to_generate)>0)

def get_drift_cost(before, after, empty_reward=True):
    condition = sum(after)-sum(before)
    before = np.sqrt(before)
    after = np.sqrt(after)
    # if sum(after)>sum(before):
    if condition>0:
        drift = after
    elif empty_reward:
        if np.all(after==0):
            drift = after-2*before
        else:
            drift=after-before
        # drift = after-before-(after==0)*before
    else:
        drift = after-before
    return drift

def get_drift_cost2(before, after, empty_reward=True):
    condition = after-before
    # mask = (condition>0)
    drift_cost = (condition>0)*after
    if empty_reward:
        drift_cost += (condition==0)*condition
    drift_cost += (condition<=0)*condition
    return drift_cost

def total_cost(used_edge_cpus, used_cloud_cpus, drift_cost, option=1):
    local_cost = 0
    server_cost = 0
    for used_edge_cpu in used_edge_cpus.values():
        # local_cost += 10*(used_edge_cpu/10)**3
        local_cost += get_local_power_cost(used_edge_cpu)
    for used_cloud_cpu in used_cloud_cpus.values():
        server_cost += 54*(used_cloud_cpu/54)**3
    #
    # print("----------------edge----------------")
    # print(used_edge_cpus)
    # print(local_cost)
    # print("----------------cloud----------------")
    # print(used_cloud_cpus)
    # print(server_cost)
    # print("----------------drift----------------")
    # print(drift_cost)
    # print("----------------3:1----------------")
    # print(aggregate_cost1(local_cost, server_cost, drift_cost))
    # print("----------------1:3----------------")
    # print(aggregate_cost2(local_cost, server_cost, drift_cost))
    # print("----------------1:1----------------")
    # print(aggregate_cost0(local_cost, server_cost, drift_cost))
    # import pdb; pdb.set_trace()

    if option==1:
        return aggregate_cost1(local_cost, server_cost, drift_cost)
    elif option==2:
        return aggregate_cost2(local_cost, server_cost, drift_cost)
    else:
        return aggregate_cost0(local_cost, server_cost, drift_cost)

# show graph 5th option
def aggregate_cost0(local_cost, server_cost, quad_drift):
    local_cost /= 900000*64*GHZ**3 #dollars/sec
    server_cost /= 900000*64*GHZ**3 #dollars/sec
    compute_cost = 2*(local_cost + server_cost)
    # drift cost와 dollar cost가 똑같이 1초에 1달러의 가치를 지닌다고 가정하는 것.

    drift_cost = 2*sum(quad_drift)

    return compute_cost+drift_cost

def aggregate_cost1(local_cost, server_cost, quad_drift):
    local_cost /= 900000*64*GHZ**3 #dollars/sec
    server_cost /= 900000*64*GHZ**3 #dollars/sec
    compute_cost = (3*local_cost/2 + 1*server_cost/2) # 190210 16:10부터
    # drift cost와 dollar cost가 똑같이 1초에 1달러의 가치를 지닌다고 가정하는 것.

    drift_cost = sum(quad_drift)# 190210 16:10부터 *2 없어짐
    # import pdb; pdb.set_trace()

    # return compute_cost+drift_cost/2
    return compute_cost+drift_cost/3 # 20200309

def aggregate_cost2(local_cost, server_cost, quad_drift):
    local_cost /= 900000*64*GHZ**3 #dollars/sec
    server_cost /= 900000*64*GHZ**3 #dollars/sec
    compute_cost = (1*local_cost/2 + 3*server_cost/2) # 190210 16:10부터
    # drift cost와 dollar cost가 똑같이 1초에 1달러의 가치를 지닌다고 가정하는 것.

    drift_cost = sum(quad_drift) # 190210 16:10부터 *2 없어짐

    # return compute_cost+drift_cost/2
    return compute_cost+drift_cost/3 # 20200309
