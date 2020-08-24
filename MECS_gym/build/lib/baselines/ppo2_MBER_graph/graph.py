import numpy as np
import matplotlib.pyplot as plt
ENV = ['Hopper','HalfCheetah','Humanoid']
DIM = [3,6,17]

COLOR = ['C0','C1','C2']


plt.figure(figsize=(3.5,3))
for i in range(len(ENV)):

    batchIS=np.loadtxt('/home/han/Downloads/log_ppo/graphdata/batchIS_%s.txt'%ENV[i])
    time_step = range(len(batchIS))
    plt.plot(time_step, batchIS[:,-1], color=COLOR[i], label='%s (dim.=%d)'% (ENV[i],DIM[i]))
plt.plot(time_step, 1.2*np.ones(len(time_step)), color = 'black', label='clipping factor')
plt.xlabel("Iterations")
plt.ylabel("Importance Sampling Weight")
# plt.legend()
plt.tight_layout()
# plt.ylim(1, 2)
plt.grid(True)
plt.savefig('/home/han/Downloads/log_ppo/graphdata/AISW.pdf')
plt.close()

plt.figure(figsize=(3.5,3))
for i in range(len(ENV)):
    notusefrac=np.loadtxt('/home/han/Downloads/log_ppo/graphdata/notuse_%s.txt'%ENV[i])
    time_step = range(len(notusefrac))
    plt.plot(time_step, 100 * notusefrac[:,-1], color=COLOR[i], label='%s (dim.=%d)'% (ENV[i],DIM[i]))
plt.plot(time_step, np.nan * notusefrac[:,-1], color='black', label='PPO clipping factor')
plt.xlabel("Iterations")
plt.ylabel("Vanishing Gradient Ratio (%)")
plt.legend()
plt.tight_layout()
# plt.ylim(0, 1)
plt.grid(True)
plt.savefig('/home/han/Downloads/log_ppo/graphdata/VGR.pdf')