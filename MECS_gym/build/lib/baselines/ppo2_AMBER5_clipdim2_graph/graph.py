import numpy as np
import matplotlib.pyplot as plt
ENV = ['Humanoid','Hopper','HalfCheetah',]
DIM = [17,3,6,]
env = ENV[0]
dim = DIM[0]
COLOR = ['C0','C1','C2','C3','C4','C5','C6']
dtarg = [0.001,0.005,0.02]


plt.figure(figsize=(3.5,3))
for i in range(len(dtarg)):
    batchIS = np.loadtxt('/home/han/Downloads/log_ppo/graphdata/batchIS_DISC_%s_%s.txt' % (str(dtarg[i]), env))
    time_step = range(len(batchIS))
    plt.plot(time_step, batchIS[:,-1], color=COLOR[i], label='$J_{targ}=%s$'% (str(dtarg[i])))
plt.xlabel("Iterations")
plt.ylabel("Importance Sampling Weight")
plt.legend()
plt.tight_layout()
plt.ylim(1, 1.5)
plt.grid(True)
plt.savefig('/home/han/Downloads/log_ppo/graphdata/AISW_DISC.pdf')
plt.close()
plt.figure(figsize=(3.5,3))
for i in range(len(dtarg)):
    batchISdim=np.loadtxt('/home/han/Downloads/log_ppo/graphdata/batchISdim_DISC_%s_%s.txt'%(str(dtarg[i]),env))
    time_step = range(len(batchISdim))
    plt.plot(time_step, batchISdim[:, -1], color=COLOR[i], label='$J_{targ}=%s$' % (str(dtarg[i])))
plt.xlabel("Iterations")
plt.ylabel("Importance Sampling Weight")
plt.legend()
plt.tight_layout()
# plt.ylim(1, 2)
plt.grid(True)
plt.savefig('/home/han/Downloads/log_ppo/graphdata/AISWdim_DISC.pdf')
plt.close()
#
# plt.figure(figsize=(3.5,3))
# for i in range(len(dtarg)):
#     notusefrac=np.loadtxt('/home/han/Downloads/log_ppo/graphdata/notuse_DISC_%s_%s.txt'%(str(dtarg[i]),env))
#     time_step = range(len(notusefrac))
#     plt.plot(time_step, notusefrac[:,-1], color=COLOR[i], label='$J_{targ}=%s$'% (str(dtarg[i])))
# plt.xlabel("Iterations")
# plt.ylabel("Vanishing Gradient Ratio (%)")
# plt.legend()
# plt.tight_layout()
# # plt.ylim(0, 1)
# plt.grid(True)
# plt.savefig('/home/han/Downloads/log_ppo/graphdata/VGR_DISC.pdf')