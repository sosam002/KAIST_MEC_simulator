import matplotlib.pyplot as plt
import numpy as np


# # 여기는 epsilon clip이 0.1
a = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_empty_reward.npy")
b = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_empty_reward_1000.npy")
c = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval.npy")
d = np.load("ppo_fixed_len2019-08-29 10:44:41.135919/eval_1000.npy")

l12 = plt.plot(a.mean(axis=1), label="empty reward", color='r')
l2 = plt.fill_between(range(len(a)), a.min(axis=1), a.max(axis=1),color='r', alpha=0.3)
l13 = plt.plot(b.mean(axis=1), label="empty reward 1000", color='b')
l3 = plt.fill_between(range(len(b)), b.min(axis=1), b.max(axis=1),color='b', alpha=0.3)
l122 = plt.plot(c.mean(axis=1), label="no empty reward", color='g')
l22 = plt.fill_between(range(len(c)), c.min(axis=1), c.max(axis=1),color='g', alpha=0.3)
l132 = plt.plot(d.mean(axis=1), label="no empty reward 1000", color='purple')
l32 = plt.fill_between(range(len(d)), d.min(axis=1), d.max(axis=1),color='purple', alpha=0.3)


# 여기는 epsilon clip이 0.2
ea = np.load("ppo_fixed_len2019-08-29 16:03:13.198024/eval_empty_reward.npy")
eb = np.load("ppo_fixed_len2019-08-29 16:03:13.198024/eval_empty_reward_1000.npy")
ec = np.load("ppo_fixed_len2019-08-29 16:03:13.198024/eval.npy")
ed = np.load("ppo_fixed_len2019-08-29 16:03:13.198024/eval_1000.npy")

l12 = plt.plot(ea.mean(axis=1), label="empty reward", color='r')
l2 = plt.fill_between(range(len(ea)), ea.min(axis=1), ea.max(axis=1),color='r', alpha=0.3)
l13 = plt.plot(eb.mean(axis=1), label="empty reward 1000", color='b')
l3 = plt.fill_between(range(len(eb)), eb.min(axis=1), eb.max(axis=1),color='b', alpha=0.3)
l122 = plt.plot(ec.mean(axis=1), label="no empty reward", color='g')
l22 = plt.fill_between(range(len(ec)), ec.min(axis=1), ec.max(axis=1),color='g', alpha=0.3)
l132 = plt.plot(ed.mean(axis=1), label="no empty reward 1000", color='purple')
l32 = plt.fill_between(range(len(ed)), ed.min(axis=1), ed.max(axis=1),color='purple', alpha=0.3)

plt.legend()
# plt.xticks([50,100,150,200,400,600,800,1000,1200,1600])
plt.grid(True, axis='x')
# plt.show([l1,l2,l3,l4])
plt.show()
