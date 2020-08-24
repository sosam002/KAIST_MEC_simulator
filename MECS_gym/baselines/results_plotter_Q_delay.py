import os
import csv
import numpy as np
import matplotlib
# matplotlib.use('Agg') # Can cwisrlge to 'Agg' for non-interactive mode
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'

# COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
#         'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
#         'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
# COLORS = [ 'orange', 'red', 'lime', 'darkblue', 'purple']

import json
import argparse
from baselines.rl_agents.results_list import *
from collections import defaultdict
ENV_NAME = ['InvertedPendulum-v1',          #0
            'InvertedDoublePendulum-v1',    #1
            'Reacher-v1',                   #2
            'HalfCheetah-v1',               #3
            'Swimmer-v1',                   #4
            'Hopper-v1',                    #5
            'Walker2d-v1',                  #6
            'Ant-v1',                       #7
            'Humanoid-v1',                 #8
            'HumanoidStandup-v1']           #9

ENV_INFO = [
    # {'env_name': 'MECS-v3_c10000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c50000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c100000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c120000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c140000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c160000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c180000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c200000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c220000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c240000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c260000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c280000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c300000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c320000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c340000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c360000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c380000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c400000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c500000.0_f1_s0', 'max_timesteps': int(1e8)},

    # {'env_name': 'MECS-v3_c10000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c50000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c100000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c150000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c200000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c250000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c300000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c400000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v3_c500000.0_f1_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c20_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c40_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c60_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c80_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c100_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c200_f0_s0', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c400_f0_s0', 'max_timesteps': int(1e8)},

    #
    {'env_name': 'MECS-v6_c20', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v6_c40', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v6_c60', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v6_c80', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v6_c100', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c200', 'max_timesteps': int(1e8)},
    # {'env_name': 'MECS-v6_c400', 'max_timesteps': int(1e8)},


    ]


ALGORITHM_INFO = [
    {'algorithm_name': 'ppo3', 'num_actors': 1, 'name_plot': 'PPO'},         #0
    {'algorithm_name': 'DISC', 'num_actors': 1, 'name_plot': 'PPO'},         #0
# {'algorithm_name': 'MPE_PPO_update5_TRatio20', 'num_actors': 4, 'name_plot': 'MPE_PPO_20'},
# {'algorithm_name': 'MPE_SAC_NA4_NQ2_update1_TRatio2_ver3', 'num_actors': 4, 'name_plot': 'IPE-SAC'},    #1
]

# COLORS = ['xkcd:orange', 'xkcd:purple' ,'xkcd:sienna', 'xkcd:tomato', 'xkcd:olive', 'xkcd:blue', 'lime'] # 16, 17, 0, 1, 2, 23

  # 22, 5 ,7, 4, 23
# COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'purple', 'black', 'tan',
#         'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
#         'darkgreen', 'darkblue']
# COLORS = ['xkcd:sienna', 'xkcd:tomato', 'xkcd:olive', 'xkcd:blue']   # 5 ,7, 4, 23
# COLORS = ['xkcd:purple', 'xkcd:tomato', 'xkcd:blue'] # 19, 18, 21

# COLORS = ['xkcd:sienna', 'xkcd:tomato', 'xkcd:olive', 'xkcd:blue', 'lime']
# COLORS = ['xkcd:orange', 'xkcd:darkgreen', 'xkcd:blue']
# COLORS = ['xkcd:tomato', 'xkcd:olive', 'xkcd:blue']
LINE_STYLES = ['-', '--', '-.', ':', 'steps']


def load_progress(dir):
    result = {}
    new_rows = []  # a holder for our modified rows when we make them
    cwisrlges = {  # a dictionary of cwisrlges to make, find 'key' substitue with 'value'
        ', ': '',  # I assume both 'key' and 'value' are strings
    }
    # with open(dir, 'r') as f:
    #     read = f.readlines()
    #     for row in read:  # iterate over the rows in the file
    #         row=row.replace(', ',' ')
    #         new_rows.append(row)  # add the modified rows
    # with open(dir, 'w') as f:
    #     # Overwrite the old file with the modified rows
    #     for row in new_rows:
    #         f.write(row)

    with open(dir, 'r') as csvfile:
        for i, row in enumerate(csv.DictReader(csvfile)):
            if i == 0:
                for key in row.keys():
                    result[key] = [maycwisrlge_str2float(row[key])]
            else:
                for key in row.keys():
                    if key is not None:
                        result[key].append(maycwisrlge_str2float(row[key]))

    return result

def maycwisrlge_str2float(str):
    try:
        return float(str)
    except:
        return str

def rolling_window(x, window_size=10):
    x_t = np.concatenate((np.array(x), np.nan+np.zeros(window_size-1)))
    x_t2 = np.mean([np.roll(x_t, i) for i in range(window_size)], axis=0)
    return x_t2[:len(x)]

def get_performance(performance, timestep_algorithm, max_timesteps=1e6, min_step=4000):
    performance = np.array(performance)
    max_t = min(timestep_algorithm[-1], max_timesteps)
    performance_t = []
    for i in range(int(max_t / min_step)):
        perf_t = performance[np.array(timestep_algorithm) <= (i+1) * min_step][-1]
        performance_t.append(perf_t)
    return np.array(performance_t), (np.arange(int(max_t / min_step)) + 1) * min_step

def plot_learning_curves(env_num,base_dir, env_name, arr_algorithm_name, num_actors, total_iter, colors=None,
                         max_timesteps=1e6, with_std=False, with_eval=False, arr_name_plot=None, with_title=True, graph_type=0, with_all_actors=False,
                         fig_size = (6, 4.5), save_filename='Performance', save_fig=True, save_format='pdf',xscale=1e6, min_step = 20000):
    filename = save_filename

    time_step = (np.arange(0, int(max_timesteps / min_step) + 1) + 1) * min_step
    # plt.figure(figsize=fig_size)
    arr_plot = []

    if colors is None:
        COLORS = ['C0', 'C1', 'C2','C3','C4','C5','C6','C7','C8','C9','black','red','blue']
    else:
        COLORS = colors
    if ('base' in filename):
        COLORS = ['C4', 'C2', 'C0', 'C1']
    if ('nov' in filename):
        COLORS = ['C0', 'C1']
    if ('clip' in filename):
        COLORS = ['C2', 'C0', 'C1', 'C4']
    if ('IStarg' in filename):
        COLORS = ['C7','C1', 'C0', 'C2','C4','C5','C6']
    if ('batchlim' in filename):
        COLORS = ['C2', 'C0', 'C1', 'C4']
    if ('other' in filename):
        COLORS = ['C0', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8',  'C1',  'C9','C10', 'C3']
    if ('amber' in filename):
        COLORS = ['C0', 'C1', 'C2']

    for i, algorithm_name in enumerate(arr_algorithm_name):
        # filename = filename + '_' + algorithm_name
        arr_queue_ = []
        arr_power_ = []
        arr_performance = []
        arr_max_performance = []

        arr_queue_t = []
        arr_power_t = []
        arr_performance_t = []
        arr_queue = []
        arr_power = []
        arr_perf = []
        arr_timestep_t = []
        if (env_name == 'Ant-v1') & ('ppo' in algorithm_name) & ('kloldnew' not in algorithm_name):
            algorithm_name += '_numt5e6ant'

        if env_name == 'HalfCheetah-v1':
            env_name_ = 'half-cheetah'
        elif env_name == 'Hopper-v1':
            env_name_ = 'hopper'
        elif env_name == 'Walker2d-v1':
            env_name_ = 'walker'
        elif env_name == 'Ant-v1':
            env_name_ = 'ant'
        elif env_name == 'Humanoid-v1':
            env_name_ = 'humanoid-gym'
        elif env_name == 'HumanoidStandup-v1':
            env_name_ = 'humanoid-standup-gym'
        elif env_name == 'BipedalWalker-v2':
            env_name_ = 'bi'
        elif env_name == 'BipedalWalkerHardcore-v2':
            env_name_ = 'bihard'

        for iter in range(total_iter):
            dir_log = os.path.join(base_dir, algorithm_name, env_name, 'iter'+str(iter), 'log.txt')
            if os.path.isfile(dir_log):
                os.remove(dir_log)
            dir = os.path.join(base_dir, algorithm_name, env_name, 'iter' + str(iter), 'progress.csv')
            result = load_progress(dir)
            # print(result)
            if  algorithm_name=='trpo_mpi':
                time_t = np.array(result['TimestepsSoFar']) * num_actors[i]
            else:
                time_t = np.array(result['total_timesteps']) * num_actors[i]
            # arr_timestep_t.append(time_step)
            arr_queue_actors = []
            arr_power_actors = []
            arr_performance_actors = []
            for actor_num in range(num_actors[i]):
                if with_eval:
                    queue_t = result['eval_edge_queue_avg']
                    power_t = result['eval_power']
                    if graph_type==1:
                        perf_t = result['eprewmean']
                    else:
                        perf_t = result['policy_entropy']

                    # q_1 = [result['eval_q_1_%02d%02d'%(i,i+1)] for i in range(50)]
                    # q_2 = [result['eval_q_2_%02d%02d'%(i,i+1)] for i in range(50)]
                    # q_3 = [result['eval_q_3_%02d%02d'%(i,i+1)] for i in range(50)]

                else:
                    queue_t = result['edge_queue_avg']
                    power_t = result['power']

                    if graph_type==1:
                        perf_t = result['eprewmean']
                    else:
                        perf_t = result['policy_entropy']
                queue_avg_t, timestep_t = get_performance(queue_t, time_t, max_timesteps=max_timesteps, min_step=min_step)
                power_avg_t, timestep_t = get_performance(power_t, time_t, max_timesteps=max_timesteps, min_step=min_step)
                performance_t, timestep_t = get_performance(perf_t, time_t, max_timesteps=max_timesteps, min_step=min_step)
                # print('timestep : ', len(timestep_t))
                if actor_num == 0:
                    arr_timestep_t.append(timestep_t)
                arr_queue_actors.append(queue_avg_t)
                arr_power_actors.append(power_avg_t)
                arr_performance_actors.append(performance_t)

            arr_queue_t.append(rolling_window(np.mean(arr_queue_actors, axis=0), window_size=1))
            arr_power_t.append(rolling_window(np.mean(arr_power_actors, axis=0), window_size=1))
            arr_performance_t.append(rolling_window(np.mean(arr_performance_actors, axis=0), window_size=1))
            arr_queue.append(arr_queue_actors)
            arr_power.append(arr_power_actors)
            arr_perf.append(arr_performance_actors)
            # print(arr_queue)
            # print(arr_perf)

        min_len = np.min([len(arr_timestep_t[i]) for i in range(total_iter)])
        for d in range(len(arr_perf)):
            # print(np.shape(arr_perf[d][0]))
            arr_queue[d][0]=arr_queue[d][0][:min_len]
            arr_power[d][0]=arr_power[d][0][:min_len]
            arr_perf[d][0]=arr_perf[d][0][:min_len]
        arr_perf = np.array(arr_perf).reshape((total_iter,-1))
        arr_queue = np.array(arr_queue).reshape((total_iter,-1))
        arr_power = np.array(arr_power).reshape((total_iter,-1))
        print(len(arr_queue[0]))
        print(len(arr_power[0]))
        # arr_mean_perf = np.mean(arr_perf[:,20:], axis=0)
        # arr_std_perf = np.std(arr_perf[:,20:], axis=0)
        # max_idx = np.argmax(arr_mean_perf)
        # max_average_return = arr_mean_perf[max_idx]
        # mar_std = arr_std_perf[max_idx]
        # print(algorithm_name + " , " + env_name + " : " + "%f"%max_average_return + "+-" + "%f"%mar_std)
        time_step = arr_timestep_t[0][:min_len]
        for iter in range(total_iter):
            arr_queue_.append(arr_queue_t[iter][:min_len])
            arr_power_.append(arr_power_t[iter][:min_len])
            arr_performance.append(arr_performance_t[iter][:min_len])

        arr_queue = np.array(arr_queue)
        arr_power = np.array(arr_power)
        arr_performance = np.array(arr_performance)
        # print(arr_performance.shape)

        avg_queue = np.mean(arr_queue, axis=0)
        avg_power = np.mean(arr_power, axis=0)
        avg_performance = np.mean(arr_performance, axis=0)
        std_performance = np.std(arr_performance, axis=0)

        color = COLORS[env_num]

        # plot, = plt.plot(time_step / xscale, max_performance, color=color)
        if graph_type==0:
            # plot, = plt.plot(np.log10(avg_queue), np.log10(avg_power), color=color, label=env_name)
            # plot, = plt.plot(avg_queue[120:130], avg_power[120:130], color=color, label=env_name)
            plt_q = avg_queue[-100:]
            plt_p = avg_power[-100:]
            plot, = plt.plot(plt_q, plt_p, color=color, label=env_name)
            print(np.mean(plt_q), np.mean(plt_p))
            print(np.mean(plt_q[-10:]), np.mean(plt_p[-10:]))
            arr_plot.append(plot)

            if with_title:
                plt.title(env_name)
            plt.xlabel("delay")
            plt.ylabel("cost")
        elif graph_type==1:
            plot, = plt.plot((time_step / xscale), avg_performance, color=color, label=env_name)
            # print(algorithm_name + " : " + str(avg_performance[-1]) + " +- " + str(std_performance[-1]))
            # for j in range(total_iter):
            #     print(str(j) + " : " + str(arr_performance[j][-1]))
            arr_plot.append(plot)
            if with_std:
                upper_performance = avg_performance + std_performance
                lower_performance = avg_performance - std_performance
                plt.fill_between(time_step / xscale, lower_performance, upper_performance, color=color, alpha=0.2,
                                 linestyle='None')
            plt.xlim(0, max_timesteps / xscale)
            # plt.ylim(-130, 100)
            if with_title:
                plt.title(env_name)

            plt.xlabel("Time Steps (1e6)")
            plt.ylabel("Average Episode Reward Sum")
        elif graph_type==2:
            plot, = plt.plot((time_step / xscale), avg_performance, color=color, label=env_name)
            # print(algorithm_name + " : " + str(avg_performance[-1]) + " +- " + str(std_performance[-1]))
            # for j in range(total_iter):
            #     print(str(j) + " : " + str(arr_performance[j][-1]))
            arr_plot.append(plot)
            if with_std:
                upper_performance = avg_performance + std_performance
                lower_performance = avg_performance - std_performance
                plt.fill_between(time_step / xscale, lower_performance, upper_performance, color=color, alpha=0.2,
                                 linestyle='None')
            plt.xlim(0, max_timesteps / xscale)
            # plt.ylim(-130, 100)
            if with_title:
                plt.title(env_name)

            plt.xlabel("Time Steps (1e6)")
            plt.ylabel("Average Episode Reward Sum")

    plt.legend()
    # plt.tight_layout()
    # plt.ylim(-20, 0)
    plt.grid(True)

    for t in range(len(arr_algorithm_name)):
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001') & ('nov' in filename):
            arr_name_plot[t] = 'GAE-V'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001') & ('clip' in filename):
            arr_name_plot[t] = '$\epsilon = 0.4$'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001') & ('IStarg' in filename):
            arr_name_plot[t] = '$J_{IS}, J_{targ} = 0.001$'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001') & ('batchlim' in filename):
            arr_name_plot[t] = '$\epsilon_b = 0.1$'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001') & ('nov' in filename):
            arr_name_plot[t] = 'GAE-V'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001') & ('clip' in filename):
            arr_name_plot[t] = '$\epsilon = 0.4$'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001') & ('IStarg' in filename):
            arr_name_plot[t] = '$J_{KL}, J_{targ} = 0.001$'
        if (arr_algorithm_name[t] == 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001') & ('batchlim' in filename):
            arr_name_plot[t] = '$\epsilon_b = 0.1$'

    # if arr_name_plot is not None:
    #     plt.legend(list(reversed(arr_plot)), list(reversed(arr_name_plot)))
    # else:
    #     plt.legend(list(reversed(arr_plot)), list(reversed(arr_algorithm_name)))
    if not os.path.isdir('/home/wisrl/Downloads/log_ppo/figures'):
        os.mkdir('/home/wisrl/Downloads/log_ppo/figures')
    if not os.path.isdir('/home/wisrl/Downloads/log_ppo/figures/'+filename):
        os.mkdir('/home/wisrl/Downloads/log_ppo/figures/'+filename)
    # plt.show()



import argparse

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_ind', type=int, default=0)
    parser.add_argument('--base_dir', help = 'Base directories', default='/home/wisrl/Downloads/log_ppo/')
    parser.add_argument('--algorithm_ind', help='List of algorithms', default=[0,'_mecs'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[1,'_mecs'])
    parser.add_argument('--total-iter', type=int, default=1)
    parser.add_argument('--fig-size', help='Size of Figure (width, height)', default=(6,4.5))
    parser.add_argument('--save-fig', type=bool, default=True)
    parser.add_argument('--with-std', type=bool, default=True)
    parser.add_argument('--with-eval', type=bool, default=True)
    parser.add_argument('--with-title', type=bool, default=True)
    parser.add_argument('--with-all-actors', type=bool, default=False)
    parser.add_argument('--graph_type', type=int, default=2) # 0 : q power / 1 : rew / 2 : entropy
    parser.add_argument('--save-format', help='figure format (eps, fig, png, etc)', default="eps")
    parser.add_argument('--fstd', type=int, default=0)
    parser.add_argument('--std', type=int, default=0)

    env_num=len(ENV_INFO)
    args = parser.parse_args()
    plt.figure(figsize=args.fig_size)
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, env_num)]
    for i in range(env_num):
        args.algorithm_name = [ALGORITHM_INFO[algorithm_ind]['algorithm_name'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.num_actors = [ALGORITHM_INFO[algorithm_ind]['num_actors'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.name_plot = [ALGORITHM_INFO[algorithm_ind]['name_plot'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.env_name = ENV_INFO[i]['env_name']+"_f{}_s{}".format(args.fstd,args.std)
        args.max_timesteps = ENV_INFO[i]['max_timesteps']
        plot_learning_curves(i,args.base_dir, args.env_name, args.algorithm_name, args.num_actors, args.total_iter, colors=colors,
                             max_timesteps=args.max_timesteps, with_std=args.with_std, with_eval=args.with_eval, with_title=args.with_title, graph_type=args.graph_type, arr_name_plot=args.name_plot,
                             fig_size=args.fig_size, save_filename=args.algorithm_ind[-1], save_fig=args.save_fig, save_format=args.save_format)
    if not args.graph_type:
        # import pdb; pdb.set_trace()

        dir_names =[]
        # dir_names +=["kkt_actor_0.0001_2020-07-30 05:29:32.285269"]
        # dir_names +=["kkt_actor_0.001_2020-07-30 05:30:10.662470"]
        # dir_names +=["kkt_actor_0.001_2020-08-19 00:03:55.338870"] #*10
        # dir_names +=["kkt_actor_0.01_2020-08-19 00:00:09.935894"] #*400
        # dir_names +=["kkt_actor_0.1_2020-08-18 23:53:56.739881"] #*100
        # dir_names +=["kkt_actor_1.0_2020-08-18 23:52:02.561109"] #*100
        # dir_names +=["kkt_actor_20000.0_2020-08-18 23:50:05.288407"] # 1/1
        # dir_names +=["kkt_actor_35000.0_2020-07-30 06:18:42.971473"]
        # dir_names +=["kkt_actor_39000.0_2020-07-30 06:24:50.132909"]
        # dir_names +=["kkt_actor_100000.0_2020-08-18 23:39:09.419154"] # 1/10
        # dir_names +=["kkt_actor_1000000.0_2020-07-30 06:36:55.707468"]
        dir_names+=["kkt_actor_0.0001_2020-08-19 21:39:55.109671"]
        dir_names+=["kkt_actor_0.0001_2020-08-19 20:31:52.112062" ,"kkt_actor_0.1_2020-08-19 16:59:28.782429",
        "kkt_actor_20000.0_2020-08-19 17:12:38.193775", "kkt_actor_40000.0_2020-08-19 19:46:45.925543", "kkt_actor_100000.0_2020-08-19 20:05:21.954633",
        "kkt_actor_1000000.0_2020-08-20 01:00:04.303694",
        "kkt_actor_2000000.0_2020-07-30 06:41:38.690166",
        "kkt_actor_3000000.0_2020-07-30 06:45:14.102170",
        "kkt_actor_4000000.0_2020-07-30 06:50:44.509121",
        "kkt_actor_5000000.0_2020-07-30 06:54:01.825878",
        "kkt_actor_7000000.0_2020-07-30 07:02:34.965285",
        "kkt_actor_8000000.0_2020-07-30 07:06:04.982889",
        "kkt_actor_10000000.0_2020-07-30 07:14:25.544843",
        ]

        graph_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        length = 5000
        model_name = 0
        clrs = ['r', 'g', 'b', 'pink', 'y', 'purple', 'cyan', 'magenta', 'k'] * 2
        markers = ['o', 'v', '*', 's', '+', 'h', 'h', 'H', 'D', 'd', 'P', 'X'] * 10


        q = []
        p = []
        # import pdb; pdb.set_trace()
        clr_idx = 0
        marker_idx=0
        for graph in dir_names:

            # dir_name = "dppresults/" + dir_names[graph]
            dir_name = "/home/wisrl/Downloads/dppresults/" + graph

            with open("{}/args.json".format(dir_name), 'r') as f:
                env_info = json.load(f)
            ############## environment parameters ##############
            edge_capability = env_info["edge_cores"] * env_info["edge_single"]
            cloud_capability = env_info["cloud_cores"] * env_info["cloud_single"]
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
                edge_s = np.transpose(s)[:40].reshape(-1, 8, len(s))
                cloud_s = np.transpose(s)[40:]
                edge_queue = edge_s[2]  # shape (8, episode length)
                edge_cpu = edge_s[3]
                cloud_queue = cloud_s[2]
                cloud_cpu = cloud_s[3]
                workload = edge_s[4]

                # edge_queue_avg = edge_queue[:3].mean(axis=1)  # shape (8,)
                edge_queue_avg = edge_queue.mean(axis=1)  # shape (8,)
                edge_queue_avg = edge_queue_avg.mean()  # float
                cloud_queue_avg = cloud_queue.mean()  # float

                edge_power = 10 * (40*edge_cpu.sum(axis=0) * (10 ** 9) / 10) ** 3  # shape (5000,)
                cloud_power = 54 * (216*cloud_cpu * (10 ** 9) / 54) ** 3  # shape (5000,)

                edge_power_avg = edge_power.mean()
                cloud_power_avg = cloud_power.mean()

                power = edge_power_avg + cloud_power_avg

            # plt.figure("power-queue graph")
                powers.append(power)
                queues.append(edge_queue_avg)
                # print(power, edge_queue_avg)
                if ep==0:
                    # plt.scatter(np.log10(edge_queue_avg), np.log10(power), label="rwd. type {}, scale {}".format(cost_type, scale), color=clrs[clr_idx], marker=markers[marker_idx])
                    plt.scatter(edge_queue_avg, power, label="rwd. type {}, scale {}".format(cost_type, scale), color=clrs[clr_idx], marker=markers[marker_idx])
                else:
                    # plt.scatter(np.log10(edge_queue_avg),np.log10(power), color=clrs[clr_idx], marker=markers[marker_idx])
                    plt.scatter(edge_queue_avg,power, color=clrs[clr_idx], marker=markers[marker_idx])
            # q[draw_idx].append(edge_queue_avg)
            # p[draw_idx].append(power)

            # print(edge_queue_avg)
            # plt.scatter(np.log10(edge_queue_avg), np.log10(power), label="rwd. type {}, {}".format(cost_type, title), color=clrs[draw_idx],
            #             marker=markers[len(q[draw_idx]) - 1])

            clr_idx+=1
            marker_idx+=1
            q.append(np.mean(queues))
            p.append(np.mean(powers))

        # plt.plot(np.log10(q), np.log10(p), ls='--')
        plt.plot(q, p, ls='--')

        plt.xlabel("avg backlog")
        plt.ylabel("avg power")
        # plt.yticks(tks)
        # plt.xlim(0.00005, 0.0004)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('/home/wisrl/Downloads/log_ppo/figures/_mecs/', "q_power.pdf"))
    else:
        plt.savefig(os.path.join('/home/wisrl/Downloads/log_ppo/figures/_mecs/', "reward.pdf"))
    plt.show()

if __name__ == '__main__':
    main()
