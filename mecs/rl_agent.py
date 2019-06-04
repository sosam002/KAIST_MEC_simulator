import json
import logging
import os
import shutil

# from mecs
import scheduler, config
# from wholemap import WholeMap

from servernode_w_queue import ServerNode

from applications import *
from channels import *
from rl.utilities import *

logger = logging.getLogger(__name__)

def init_for_sosam(edge_capability, cloud_capability, channel, *applications):
    edge_server = ServerNode(edge_capability)
    cloud_server = ServerNode(cloud_capability)
    edge_server.add_link(cloud_server, channel)
    edge_server.make_application_queues(*applications)
    cloud_server.make_application_queues(*applications)
    return edge_server, cloud_server

def main():


    edge_capability = 30000
    cloud_capability = 300000  # clock per tick
    channel = WIRED
    applications = (AR, VR)
    task_rate = 10000 # 들어갈만한 다른 곳을 찾고 싶다.ㅠㅠ

    log_dir = 'result_sosam'
    mobile_log = {}
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    quad_Lyapunov_buffer = Lyapunov_buffer()
    edge_server, cloud_server = init_for_sosam(edge_capability, cloud_capability, channel, *applications)

    cloud_policy = [0.5, 0.5]

    for t in range(200001):
        edge_server.random_task_generation(task_rate, t, VR, AR)
        # 이건 진짜 arrival rate 이 아님.. arrival만 저장하는걸 또 따로 만들어야 한다니 고통스럽다.
        print(edge_server.queue_list[AR].tasks, edge_server.queue_list[VR].tasks)
        print(cloud_server.queue_list[AR].tasks, cloud_server.queue_list[VR].tasks)


        '''
        server node state (queue length)
        edge node stat (queue length)
        server node와 edge node의 cpu랑 통신 rate spec도 입력받아야할 듯
        저장해놔야 함.
        '''

        '''
        action받아오기! alpha, beta 받는 자리
        '''
        # 지금은 policy가 없으니까 내맘대로
        alpha = [0.5,0.5]
        beta = [0.5,0.5]

        used_edge_cpu = edge_server.do_tasks(alpha)
        used_tx, task_to_be_offloaded = edge_server.offload_tasks(beta, cloud_server.get_uuid())
        cloud_server.offloaded_tasks(task_to_be_offloaded, t)
        used_cloud_cpu = cloud_server.do_tasks(cloud_policy)


        # used_tx, task_to_be_offloaded = edge_server.do_and_offload( alpha, beta, cloud_server.get_uuid() )
        print("used_edge_cpu {}, used_tx {}, used_cloud_cpu {}".format(used_edge_cpu, used_tx, used_cloud_cpu))
        # import pdb; pdb.set_trace()
        '''
        used_cpu랑 offload 정보 받아서 reward 받는 자리
        '''

        '''
        server node state (queue length)
        edge node stat (queue length)
        server node와 edge node의 cpu랑 통신 rate spec도 입력받아야할 듯
        '''

        # get state
        edge_state = edge_server.get_status()
        cloud_state = cloud_server.get_status()
        estimated_arrival_rate = list(edge_server.estimate_arrival_rate())
        # 어휴
        state = estimated_arrival_rate + edge_state + cloud_state + [edge_server.get_channel_rate(cloud_server.get_uuid())]
        print("state = {}".format(state))

        # get reward
        quad_Lyapunov_buffer.add(quad_Lyapunov(edge_server.queue_list)+quad_Lyapunov(cloud_server.queue_list))
        quad_drift = quad_Lyapunov_buffer.get_drift()
        local_cost = local_energy_consumption(used_edge_cpu)
        server_cost = offload_cost(task_to_be_offloaded)
        cost = my_rewards(local_cost, server_cost, quad_drift)
        print("cost = {}".format(cost))

        '''
        네트워크에 (s,a,r,s') 집어넣고,
        '''
        # if t % 1000 == 0:
        #     logger.info(
        #         "================= << [%d,%d] second >> =================",
        #         t // 1000, t // 1000 + 1)
        #     logger.debug(json.dumps(mobile_log))
        #     mobile_log = {}
        # mobile_log[t] = my_map.simulate_one_time(t)


    # my_map.print_all_mobiles()


if __name__ == "__main__":
    config.initialize_mecs()
    main()
