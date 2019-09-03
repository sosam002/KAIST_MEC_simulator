import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--option', default = 1, type=int)


# # 여기는 epsilon clip이 0.1
# a = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_empty_reward.npy")
# b = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_empty_reward_1000.npy")
# c = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval.npy")
# d = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_1000.npy")
#
# l12 = plt.plot(a.mean(axis=1), label="empty reward", color='r')
# l2 = plt.fill_between(range(len(a)), a.min(axis=1), a.max(axis=1),color='r', alpha=0.3)
# l13 = plt.plot(b.mean(axis=1), label="empty reward 1000", color='b')
# l3 = plt.fill_between(range(len(b)), b.min(axis=1), b.max(axis=1),color='b', alpha=0.3)
# l122 = plt.plot(c.mean(axis=1), label="no empty reward", color='g')
# l22 = plt.fill_between(range(len(c)), c.min(axis=1), c.max(axis=1),color='g', alpha=0.3)
# l132 = plt.plot(d.mean(axis=1), label="no empty reward 1000", color='purple')
# l32 = plt.fill_between(range(len(d)), d.min(axis=1), d.max(axis=1),color='purple', alpha=0.3)


# # 여기는 epsilon clip이 0.2
# dir_name = "ppo_fixed_len2019-08-29 16:03:13.198024/eval_results"
# ea = np.load("{}/eval_empty_reward.npy".format(dir_name))
# eb = np.load("{}/eval_empty_reward_1000.npy".format(dir_name))
# ec = np.load("{}/eval.npy".format(dir_name))
# ed = np.load("{}/eval_1000.npy".format(dir_name))
#
# l12 = plt.plot(ea.mean(axis=1), label="empty reward", color='r')
# l2 = plt.fill_between(range(len(ea)), ea.min(axis=1), ea.max(axis=1),color='r', alpha=0.3)
# # l13 = plt.plot(eb.mean(axis=1), label="empty reward 1000", color='b')
# # l3 = plt.fill_between(range(len(eb)), eb.min(axis=1), eb.max(axis=1),color='b', alpha=0.3)
# l122 = plt.plot(ec.mean(axis=1), label="no empty reward", color='g')
# l22 = plt.fill_between(range(len(ec)), ec.min(axis=1), ec.max(axis=1),color='g', alpha=0.3)
# # l132 = plt.plot(ed.mean(axis=1), label="no empty reward 1000", color='purple')
# # l32 = plt.fill_between(range(len(ed)), ed.min(axis=1), ed.max(axis=1),color='purple', alpha=0.3)
#

# 여기는 epsilon clip이 0.2
# empty_rewards = []
# no_empty_rewards = []

dir_name = "results/ppo_fixed_len2019-09-02 16:41:42.821829"
dir_name = "results/ppo_fixed_len2019-09-03 13:07:38.099530"
a = np.load("{}/eval_results/eval_empty_reward.npy".format(dir_name))
b = np.load("{}/eval_results/eval.npy".format(dir_name))
with open("{}/args.json".format(dir_name), 'r') as f:
    args = json.load(f)
comment1=args["comment"]
dir_name = "results/ppo_fixed_len2019-09-02 16:42:11.726059"
dir_name = "results/ppo_fixed_len2019-09-03 13:08:35.067297"
c = np.load("{}/eval_results/eval_empty_reward.npy".format(dir_name))
d = np.load("{}/eval_results/eval.npy".format(dir_name))
with open("{}/args.json".format(dir_name), 'r') as f:
    args = json.load(f)
comment2=args["comment"]
dir_name = "results/ppo_fixed_len2019-09-02 16:42:26.598938"
dir_name = "results/ppo_fixed_len2019-09-03 13:08:45.010138"
e = np.load("{}/eval_results/eval_empty_reward.npy".format(dir_name))
f = np.load("{}/eval_results/eval.npy".format(dir_name))
with open("{}/args.json".format(dir_name), 'r') as file:
    args = json.load(file)
comment3=args["comment"]

args = parser.parse_args()

if args.option==1:
    # computing cost with deterministic arrival
    plt.plot(a.mean(axis=1), label="empty reward {}".format(comment1), color='r')
    plt.fill_between(range(len(a)), a.min(axis=1), a.max(axis=1),color='r', alpha=0.3)
    plt.plot(b.mean(axis=1), label="no empty reward {}".format(comment1), color='g')
    plt.fill_between(range(len(b)), b.min(axis=1), b.max(axis=1),color='g', alpha=0.3)

if args.option==2:
    # deterministic arrival
    plt.plot(c.mean(axis=1), label="empty reward {}".format(comment2), color='r')
    plt.fill_between(range(len(c)), c.min(axis=1), c.max(axis=1),color='r', alpha=0.3)
    plt.plot(d.mean(axis=1), label="no empty reward {}".format(comment2), color='g')
    plt.fill_between(range(len(d)), d.min(axis=1), d.max(axis=1),color='g', alpha=0.3)

if args.option==3:
    # normal dist. arrival
    plt.plot(e.mean(axis=1), label="empty reward {}".format(comment3), color='r')
    plt.fill_between(range(len(e)), e.min(axis=1), e.max(axis=1),color='r', alpha=0.3)
    plt.plot(f.mean(axis=1), label="no empty reward {}".format(comment3), color='g')
    plt.fill_between(range(len(f)), f.min(axis=1), f.max(axis=1),color='g', alpha=0.3)


plt.legend()
# plt.xticks([50,100,150,200,400,600,800,1000,1200,1600])
plt.grid(True, axis='x')
# plt.show([l1,l2,l3,l4])
plt.show()
