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
                        perf_t = result['eval_eprewmean']
                    else:
                        perf_t = result['policy_entropy']

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
        print(arr_queue)
        print(arr_power)
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
            plot, = plt.plot(np.log10(avg_queue), np.log10(avg_power), color=color, label=env_name)
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

    {'env_name': 'MECS-v3_c10000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c50000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c100000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c150000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c200000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c250000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c300000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c400000.0_f1_s0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c500000.0_f1_s0', 'max_timesteps': int(1e8)},


    {'env_name': 'MECS-v3_c10000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c100000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c120000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c140000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c160000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c180000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c200000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c220000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c240000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c260000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c280000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c300000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c320000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c340000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c360000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c380000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c400000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v3_c500000.0_f0_s2', 'max_timesteps': int(1e8)},

    {'env_name': 'MECS-v1_c200000.0_f0_s2', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c300000.0_f0_s2', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c400000.0_f0_s2', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c500000.0_f0_s2', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c10000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c20000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c30000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c40000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c50000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c60000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c70000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c80000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c90000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c100000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c110000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c120000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c130000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c140000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c150000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c160000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c170000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c180000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c190000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c200000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c300000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c400000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c500000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c10000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c50000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c100000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c200000.0', 'max_timesteps': int(1e8)},
    {'env_name': 'MECS-v1_c300000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c400000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c500000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c210000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c220000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c230000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c240000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c250000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c260000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c270000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c280000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c290000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c300000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c400000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c500000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c1000000.0', 'max_timesteps': int(5e7)},
    {'env_name': 'MECS-v1_c10000000.0', 'max_timesteps': int(5e7)},

    {'env_name': 'MECS-v1_c1e-05', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c0.0001', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c0.001', 'max_timesteps': int(3e7)},                  #0
    {'env_name': 'MECS-v1_c0.01', 'max_timesteps': int(3e7)},             #3
    {'env_name': 'MECS-v1_c0.1', 'max_timesteps': int(3e7)},            #4
    {'env_name': 'MECS-v1_c1.0', 'max_timesteps': int(3e7)},  # 7
    {'env_name': 'MECS-v1_c10.0', 'max_timesteps': int(3e7)},  # 8                 #1
    {'env_name': 'MECS-v1_c100.0', 'max_timesteps': int(3e7)},  # 2
    {'env_name': 'MECS-v1_c1000.0', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c10000.0', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c10000.0', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c100000.0', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c1000000.0', 'max_timesteps': int(3e7)},
    {'env_name': 'MECS-v1_c10000000.0', 'max_timesteps': int(3e7)},
    ]


# ENV_INFO = [
#     {'env_name': 'MECS-v3_c0.01', 'max_timesteps': int(5e7)},             #3
#     {'env_name': 'MECS-v3_c0.1', 'max_timesteps': int(5e7)},            #4
#     {'env_name': 'MECS-v3_c1.0', 'max_timesteps': int(5e7)},  # 7
#     {'env_name': 'MECS-v3_c10.0', 'max_timesteps': int(5e7)},  # 8                 #1
#     {'env_name': 'MECS-v3_c100.0', 'max_timesteps': int(5e7)},  # 2
#     {'env_name': 'MECS-v3_c1000.0', 'max_timesteps': int(5e7)},
#     {'env_name': 'MECS-v3_c10000.0', 'max_timesteps': int(5e7)},
#     {'env_name': 'MECS-v3_c10000.0', 'max_timesteps': int(5e7)},
#     {'env_name': 'MECS-v3_c100000.0', 'max_timesteps': int(5e7)},
#     {'env_name': 'MECS-v3_c1000000.0', 'max_timesteps': int(5e7)},
#     {'env_name': 'MECS-v3_c10000000.0', 'max_timesteps': int(5e7)},
#     ]

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

import argparse

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_ind', type=int, default=0)
    parser.add_argument('--base_dir', help = 'Base directories', default='/home/wisrl/Downloads/log_ppo/')
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[0,'_mecs'])
    parser.add_argument('--algorithm_ind', help='List of algorithms', default=[1,'_mecs'])
    parser.add_argument('--total-iter', type=int, default=1)
    parser.add_argument('--fig-size', help='Size of Figure (width, height)', default=(6,4.5))
    parser.add_argument('--save-fig', type=bool, default=True)
    parser.add_argument('--with-std', type=bool, default=True)
    parser.add_argument('--with-eval', type=bool, default=True)
    parser.add_argument('--with-title', type=bool, default=True)
    parser.add_argument('--with-all-actors', type=bool, default=False)
    parser.add_argument('--graph-type', type=int, default=2) # 0 : q power / 1 : rew / 2 : entropy
    parser.add_argument('--save-format', help='figure format (eps, fig, png, etc)', default="eps")

    env_num=5
    args = parser.parse_args()
    plt.figure(figsize=args.fig_size)
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, env_num)]
    for i in range(env_num):
        args.algorithm_name = [ALGORITHM_INFO[algorithm_ind]['algorithm_name'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.num_actors = [ALGORITHM_INFO[algorithm_ind]['num_actors'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.name_plot = [ALGORITHM_INFO[algorithm_ind]['name_plot'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.env_name = ENV_INFO[i]['env_name']
        args.max_timesteps = ENV_INFO[i]['max_timesteps']
        plot_learning_curves(i,args.base_dir, args.env_name, args.algorithm_name, args.num_actors, args.total_iter, colors=colors,
                             max_timesteps=args.max_timesteps, with_std=args.with_std, with_eval=args.with_eval, with_title=args.with_title, graph_type=args.graph_type, arr_name_plot=args.name_plot,
                             fig_size=args.fig_size, save_filename=args.algorithm_ind[-1], save_fig=args.save_fig, save_format=args.save_format)
    if not args.graph_type:


        # parser = argparse.ArgumentParser()
        # parser.add_argument('X', type=int, nargs='+')
        # parser.add_argument('--length', default=5000, type=int)
        # parser.add_argument('--model_name', default=0, metavar='G', help="select a pytorch model ",
        #                     type=int)  # clock per tick, unit=GHZ
        #
        # args = parser.parse_args()
        graph_list = [0,1,2,3,4,5,6,7,8]
        length = 5000
        model_name = 0
        clrs = ['r', 'g', 'b', 'pink', 'y', 'purple', 'cyan', 'magenta', 'k'] * 2
        markers = ['o', 'v', '*', 's', '+', 'h', 'h', 'H', 'D', 'd', 'P', 'X'] * 2
        # q = np.empty(shape=(3,12))
        # p = np.empty(shape=(3,12))
        q = defaultdict(list)
        p = defaultdict(list)
        # import pdb;
        # pdb.set_trace()
        for graph in graph_list:

            dir_name = "/home/wisrl/Downloads/mecs_/baselines/rl_agents/results/" + dir_names[graph]

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
            print("caps {}, {}, cost type {}, seed {}".format(edge_capability, cloud_capability, cost_type, seed))
            if "kkt" not in dir_name:
                a = np.load("{}/simulate/{}_actions.npy".format(dir_name, model_name))
                s = np.load("{}/simulate/{}_states.npy".format(dir_name, model_name))
                title = "DRL"
                draw_idx = 0
            else:
                a = np.load("{}/simulate/actions_0.npy".format(dir_name))
                s = np.load("{}/simulate/states_0.npy".format(dir_name))
                title = "DPP, delta L"
                draw_idx = 1
            edge_s = np.transpose(s)[:40].reshape(-1, 8, len(s))
            cloud_s = np.transpose(s)[40:]
            edge_queue = edge_s[2]  # shape (8, episode length)
            edge_cpu = edge_s[3]
            cloud_queue = cloud_s[2]
            cloud_cpu = cloud_s[3]
            workload = edge_s[4]

            edge_queue_avg = edge_queue.mean(axis=1)  # shape (8,)
            edge_queue_avg = edge_queue_avg.mean()  # float
            cloud_queue_avg = cloud_queue.mean()  # float

            edge_power = 10 * (edge_cpu.sum(axis=0) * (10 ** 9) / 10) ** 3  # shape (5000,)
            cloud_power = 54 * (cloud_cpu * (10 ** 9) / 54) ** 3  # shape (5000,)

            edge_power_avg = edge_power.mean()
            cloud_power_avg = cloud_power.mean()

            power = edge_power_avg + cloud_power_avg

            # plt.figure("power-queue graph")

            q[draw_idx].append(edge_queue_avg)
            p[draw_idx].append(power)

            # print(edge_queue_avg)
            plt.scatter(np.log10(edge_queue_avg), np.log10(power), label="rwd. type {}, {}".format(cost_type, title), color=clrs[draw_idx],
                        marker=markers[len(q[draw_idx]) - 1])

        for k in q.keys():
            plt.plot(np.log10(q[k]), np.log10(p[k]), ls='--', color=clrs[k])

        plt.xlabel("avg delay")
        plt.ylabel("avg power used")
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
