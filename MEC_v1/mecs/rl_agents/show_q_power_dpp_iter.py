import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('X', type=int, nargs='+')
parser.add_argument('--length', default = 5000, type=int)
parser.add_argument('--model_name', default = 0, metavar='G', help = "select a pytorch model ", type=int)  # clock per tick, unit=GHZ

args = parser.parse_args()
graph_list = args.X
length = args.length
model_name = args.model_name
clrs = ['r','g','b','pink','y','purple','cyan','magenta', 'k']*2
markers = ['o', 'v', '*', 's', '+', 'h', 'h', 'H', 'D', 'd', 'P', 'X']*4

dir_names=[]

# dir_names +=["kkt_actor_0.0001_2020-07-30 05:29:32.285269"]

dir_names+=["c0.0001_s100.0_d2020-08-20 21:42:12.011553", "kkt_actor_0.0001_2020-08-19 21:39:55.109671"]
dir_names+=["kkt_actor_0.0001_2020-08-19 20:31:52.112062" ,"kkt_actor_0.1_2020-08-19 16:59:28.782429", "c0.1_s100.0_d2020-08-20 17:09:17.817053",
"kkt_actor_20000.0_2020-08-19 17:12:38.193775", "c20000.0_s1.0_d2020-08-20 22:18:15.210689","kkt_actor_40000.0_2020-08-19 19:46:45.925543", "kkt_actor_100000.0_2020-08-19 20:05:21.954633",
"kkt_actor_1000000.0_2020-08-20 01:00:04.303694",
]

# q = np.empty(shape=(3,12))
# p = np.empty(shape=(3,12))
q = []
p = []
# import pdb; pdb.set_trace()
clr_idx = 0
marker_idx=0
for graph in graph_list:

    dir_name = "dppresults/" + dir_names[graph]


    with open("{}/args.json".format(dir_name), 'r') as f:
        env_info = json.load(f)
    ############## environment parameters ##############
    edge_capability = env_info["edge_cores"]*env_info["edge_single"]
    cloud_capability =env_info["cloud_cores"]*env_info["cloud_single"]
    if edge_capability < 1e4:
        edge_capability *= 1e9
        cloud_capability *= 1e9
    cost_type = env_info["cost_type"]
    seed = env_info["random_seed"]
    number_of_apps = env_info["number_of_apps"]
    try:
        scale = env_info["scale"]
        iter = env_info["iter"]
    except:
        scale = 1
        iter = 1
    print("caps {}, {}, cost type {}, scale {}".format(edge_capability, cloud_capability, cost_type, scale))
    powers=[]
    queues=[]
    for ep in range(iter):
        a = np.load("{}/simulate/actions_{}.npy".format(dir_name,ep))
        s = np.load("{}/simulate/states_{}.npy".format(dir_name,ep))

        edge_s = np.transpose(s)[:40].reshape(-1,8,len(s))
        cloud_s = np.transpose(s)[40:]
        edge_queue = edge_s[2] # shape (8, episode length)
        edge_cpu = edge_s[3]
        cloud_queue = cloud_s[2]
        cloud_cpu = cloud_s[3]
        workload = edge_s[4]

        edge_queue_avg = edge_queue[:3].mean(axis=1) # shape (8,)
        edge_queue_avg = edge_queue_avg.mean() # float
        cloud_queue_avg = cloud_queue.mean() # float

        edge_power = 10*(40*edge_cpu.sum(axis=0)*(10**9)/10)**3 # shape (5000,)
        cloud_power = 54*(216*cloud_cpu*(10**9)/54)**3 # shape (5000,)

        edge_power_avg = edge_power.mean()
        cloud_power_avg = cloud_power.mean()

        power = edge_power_avg + cloud_power_avg


        plt.figure("power-queue graph")

        powers.append(power)
        queues.append(edge_queue_avg)
        print(power, edge_queue_avg)
        if ep==0:
            plt.scatter(edge_queue_avg,power, label="rwd. type {}, scale {}".format(cost_type, scale), color=clrs[clr_idx], marker=markers[marker_idx])
        else:
            plt.scatter(edge_queue_avg,power, color=clrs[clr_idx], marker=markers[marker_idx])

    clr_idx +=1
    marker_idx+=1

    q.append(np.mean(queues))
    p.append(np.mean(powers))

plt.plot(q,p, ls='--')

plt.xlabel("avg delay")
plt.ylabel("avg power used")
# plt.yticks(tks)
# plt.xlim(0.00005, 0.0004)
plt.legend()
plt.grid(True)
plt.show()
