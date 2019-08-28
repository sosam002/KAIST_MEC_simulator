import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

c = np.load("ppo22019-08-23 13:42:53.428830/eval_100steps.npy") #2.5, 2.5 not admissible, beta=0.5, ppo2
d = np.load("ppo22019-08-23 13:42:53.428830/eval.npy") #2.5, 2.5 not admissible, beta=0.5, ppo2
e = np.load("ppo22019-08-23 13:43:01.271534/eval_100steps.npy") # admissible, beta=1, ppo2
f = np.load("ppo22019-08-23 13:43:01.271534/eval.npy") # admissible, beta=1, ppo2
g = np.load("ppo2_fixed_len2019-08-23 14:10:18.157744/eval.npy") # admissible, empty queue reward=1, beta=0.5, fixed ppo
h = np.load("ppo2_fixed_len2019-08-23 14:59:53.182524/eval.npy") # admissible, empty queue reward=1, beta=1, fixed ppo
i = np.load("ppo3_2019-08-23 14:59:57.126770/eval_100steps.npy") # admissible, empty queue reward=1, beta=1, ppo3, network dim.
j = np.load("ppo3_2019-08-23 14:59:57.126770/eval.npy") # admissible, empty queue reward=1, beta=1, ppo3, network dim.



l6 = plt.plot(c, label = "1000steps, not admissible")
l7 = plt.plot(d, label = "steps varies, not admissible")
l8 = plt.plot(e, label = "1000steps, beta=1")
l9 = plt.plot(f, label = "steps varies, beta=1")
l11 =plt.plot(h, label = "ppo_fixed_len.py, beta=1, empty reward")
l10 =plt.plot(g, label = "ppo_fixed_len.py, beta=0.5, empty reward")

l12 = plt.plot(i, label = "ppo3_1000steps, beta=1, empty reward")
l13 = plt.plot(j, label = "ppo3_steps varies, beta=1, empty reward")

plt.legend()
# plt.xticks([50,100,150,200,400,600,800,1000,1200,1600])
plt.grid(True, axis='x')
# plt.show([l1,l2,l3,l4])
plt.show()
