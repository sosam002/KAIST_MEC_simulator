import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from baselines.bench.monitor import load_results

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'average reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y


def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def plot_curves(xy_lists, xaxis, yaxis, title, log_dir, agent_infos, time_steps):
    # fig = plt.figure(figsize=(8, 2))
    fig = plt.figure()
    y_last = []
    for i in range(len(xy_lists)):
        xy_list = xy_lists[i]
        maxx = max(xy[0][-1] for xy in xy_list)
        minx = 0
        color = COLORS[i]
        y_all = []
        # for j in range(len(xy_list)):
        for (j, (x, y)) in enumerate(xy_list):
            # plt.scatter(x, y, s=2)
            x_, y_ = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
            y_all.append(y_[find_closest(x_,time_steps)])
        y_mean = np.mean(y_all,axis=0)
        idx = 0
        for k in range(len(y_mean)):
            if y_mean[k]==y_mean[0]:
                idx=k
        y_std = np.std(y_all,axis=0)
        plt.plot(time_steps[idx:], y_mean[idx:], color=color, label=agent_infos[i])
        plt.fill_between(time_steps[idx:], y_mean[idx:]-y_std[idx:],y_mean[idx:]+y_std[idx:], color=color, alpha=0.3)
        y_last.append(y_mean[-1])
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)
    plt.legend()
    fig_name = log_dir + '/' + title + '_ylast:'
    for i in range(len(agent_infos)):
        fig_name += agent_infos[i] +'_%.2f_'% y_last[i]

    print(fig_name)
    plt.savefig(fig_name, format = 'pdf')

def plot_results(log_dir, dirs, num_timesteps, xaxis, yaxis, task_name, agent_infos):
    time_step = 1000
    time_steps = np.arange(0, int(num_timesteps / time_step)) * time_step + time_step
    # print(time_steps)
    xy_lists = []
    for dir in dirs:
        tslist = []
        for iter_ in dir:
            ts = load_results(iter_)
            ts = ts[ts.l.cumsum() <= num_timesteps]
            tslist.append(ts)
        # print(tslist)
        xy_lists.append([ts2xy(ts, xaxis, yaxis) for ts in tslist])
    plot_curves(xy_lists, xaxis, yaxis, task_name, log_dir, agent_infos, time_steps)


# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files





def main():
    import argparse
    import os
    Mujoco_Envs = ['HalfCheetah-v1', 'Hopper-v1', 'InvertedDoublePendulum-v1', 'InvertedPendulum-v1', 'Swimmer-v1',
                   'Reacher-v1', 'Walker2d-v1', 'Pendulum-v0', 'BipedalWalker-v2',
                   'BipedalWalkerHardcore-v2', 'Humanoid-v1', 'HumanoidStandup-v1', 'Ant-v1', ]
    # Mujoco_Envs = ['HalfCheetah-v1', 'Reacher-v1',]
    clip = 0.2
    dtarg = 0.001
    print(str(dtarg))
    Algorithms = ['ppo2_MBER_leng1_clip0.2', #0
                  'ppo2_MBER_leng2_clip%.1f'%clip,
                  'ppo2_MBER_leng4_clip%.1f' % clip,
                  'ppo2_MBER_leng8_clip%.1f' % clip,
                  'ppo2_MBER_leng1_clip%.1f_adaptive_clip_oldkl' % clip, #4
                  'ppo2_MBER_leng2_clip%.1f_adaptive_clip_oldkl' % clip,
                  'ppo2_MBER_leng4_clip%.1f_adaptive_clip_oldkl' % clip,
                  'ppo2_MBER_leng8_clip%.1f_adaptive_clip_oldkl' % clip,
                  'ppo2_MBER2_leng1_clip%.1f_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)), #8
                  'ppo2_MBER2_leng2_clip%.1f_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER2_leng4_clip%.1f_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER2_leng8_clip%.1f_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER3_leng1_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)), #12
                  'ppo2_MBER3_leng2_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER3_leng4_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER3_leng8_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER4_leng1_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)), #16
                  'ppo2_MBER4_leng2_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER4_leng4_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER4_leng8_clip%.1f_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg)),
                  'ppo2_MBER4_clipdim2_leng1_clip%s_vtrace_adaptive_approx_kl_dtarg%s' % (str(clip), str(dtarg)),  #20
                  'ppo2_MBER4_clipdim2_leng2_clip%s_vtrace_adaptive_approx_kl_dtarg%s' % (str(clip), str(dtarg)),
                  'ppo2_MBER4_clipdim2_leng4_clip%s_vtrace_adaptive_approx_kl_dtarg%s' % (str(clip), str(dtarg)),
                  'ppo2_MBER4_clipdim2_leng8_clip%s_vtrace_adaptive_approx_kl_dtarg%s' % (str(clip), str(dtarg)),
                  ]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', type=str, default=['/home/han/Downloads/log_server_ppo/Mujoco/'
                                                                                     + Algorithms[i] for i in [16,17,18,19]])
    parser.add_argument('--num_repeats', type=int, default=5)
    parser.add_argument('--num_timesteps', type=int, default=int(3e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--yaxis', help = 'Varible on Y-axis', default = Y_REWARD)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'Breakout')
    args = parser.parse_args()
    # args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    # agent_infos=['PPO','PPO-CONST']
    agent_infos = ['PPO-MBER(L=1)','PPO-MBER(L=2)','PPO-MBER(L=4)','PPO-MBER(L=8)']
    # agent_infos=['PPO']
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER1LengthCompare_clip%.1f'%clip
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER1LengthCompare_adaptive_oldkl_clip%.1f' % clip
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER2LengthCompare_clip%.1f_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg))
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER3LengthCompare_clip%.1f_vtrace_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg))
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER4LengthCompare_clip%.1f_vtrace_adaptive_approx_kl_dtarg%s' % (clip, str(dtarg))
    log_dir = '/home/han/Downloads/log_server_ppo/Mujoco_figures/MBER4LengthCompare_clip%s_vtrace_useadv_adaptive_approx_kl_dtarg%s' % (str(clip), str(dtarg))
    # log_dir = './plot'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    for i in range(len(Mujoco_Envs)):
        dirs = [args.dirs[j] + '/%s'%Mujoco_Envs[i] for j in range(len(args.dirs))]
        for j in range(len(dirs)):
            dirs[j]=[dirs[j]+'/iter%d'%k for k in range(args.num_repeats)]
        plot_results(log_dir, dirs, args.num_timesteps, args.xaxis, args.yaxis, Mujoco_Envs[i], agent_infos)
    # plt.show()

if __name__ == '__main__':
    main()
