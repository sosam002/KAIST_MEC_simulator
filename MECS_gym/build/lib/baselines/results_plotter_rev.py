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




def load_progress(dir):
    result = {}
    new_rows = []  # a holder for our modified rows when we make them
    cwisrlges = {  # a dictionary of cwisrlges to make, find 'key' substitue with 'value'
        ', ': '',  # I assume both 'key' and 'value' are strings
    }
    with open(dir, 'r') as f:
        read = f.readlines()
        for row in read:  # iterate over the rows in the file
            row=row.replace(', ',' ')
            new_rows.append(row)  # add the modified rows
    with open(dir, 'w') as f:
        # Overwrite the old file with the modified rows
        for row in new_rows:
            f.write(row)

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
    x_t = np.concatenate((np.array(x), np.zeros(window_size-1)))
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

def plot_learning_curves(base_dir, env_name, arr_algorithm_name, num_actors, total_iter,
                         max_timesteps=1e6, with_std=False, with_max_performance=False, arr_name_plot=None, with_title=True, with_all_actors=False,
                         fig_size = (6, 4.5), save_filename='Performance', save_fig=True, save_format='pdf',xscale=1e6, min_step = 4000):
    filename = save_filename
    if with_max_performance:
        filename = "Max_" + filename

    time_step = (np.arange(0, int(max_timesteps / min_step) + 1) + 1) * min_step
    plt.figure(figsize=fig_size)
    arr_plot = []


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
        arr_performance = []
        arr_max_performance = []

        arr_performance_t = []
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
            if algorithm_name == 'SAC' or algorithm_name == 'test_TD3_ver2_NQ2' or algorithm_name == 'SQL':

                log_dir = base_dir + env_name_ + '/' + algorithm_name + '/' + 'iter' + str(iter + 1)
                if not os.path.isdir(log_dir):
                    if iter == 4:
                        iter = 5
                    elif iter == 3:
                        iter = 6
                    elif iter == 2:
                        iter = 7
                # print(log_dir)
                dir = os.path.join(base_dir, env_name_, algorithm_name, 'iter' + str(iter + 1), 'progress.csv')
                result = load_progress(dir)

                if 'total-samples' in result:
                    time_t = np.array(result['total-samples']) * num_actors[i]
                else:
                    time_t = np.array(result['total-samples/0']) * num_actors[i]
                # arr_timestep_t.append(time_step)

                arr_performance_actors = []
                for actor_num in range(num_actors[i]):
                    if 'return-average' in result and num_actors[i] == 1:
                        perf_t = result['return-average']
                    else:
                        perf_t = result['return-average/{i}'.format(i=actor_num)]

                    performance_t, timestep_t = get_performance(perf_t, time_t, max_timesteps=max_timesteps,
                                                                min_step=min_step)
                    # print('timestep : ', len(timestep_t))
                    if actor_num == 0:
                        arr_timestep_t.append(timestep_t)
                    arr_performance_actors.append(performance_t)

            else:
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

                arr_performance_actors = []
                for actor_num in range(num_actors[i]):
                    perf_t = result['eval_eprewmean']

                    performance_t, timestep_t = get_performance(perf_t, time_t, max_timesteps=max_timesteps, min_step=min_step)
                    # print('timestep : ', len(timestep_t))
                    if actor_num == 0:
                        arr_timestep_t.append(timestep_t)
                    arr_performance_actors.append(performance_t)

            arr_performance_t.append(rolling_window(np.mean(arr_performance_actors, axis=0)))
            arr_perf.append(arr_performance_actors)

        min_len = np.min([len(arr_timestep_t[i]) for i in range(total_iter)])
        for d in range(len(arr_perf)):
            # print(np.shape(arr_perf[d][0]))
            arr_perf[d][0]=arr_perf[d][0][:min_len]
        arr_perf = np.array(arr_perf).reshape((total_iter,-1))
        arr_mean_perf = np.mean(arr_perf[:,20:], axis=0)
        arr_std_perf = np.std(arr_perf[:,20:], axis=0)
        max_idx = np.argmax(arr_mean_perf)
        max_average_return = arr_mean_perf[max_idx]
        mar_std = arr_std_perf[max_idx]
        print(algorithm_name + " , " + env_name + " : " + "%f"%max_average_return + "+-" + "%f"%mar_std)
        time_step = arr_timestep_t[0][:min_len]
        for iter in range(total_iter):
            # print(len(arr_performance_t[iter][:min_len]))
            arr_performance.append(arr_performance_t[iter][:min_len])

        arr_performance = np.array(arr_performance)
        # print(arr_performance.shape)
        avg_performance = np.mean(arr_performance, axis=0)
        std_performance = np.std(arr_performance, axis=0)

        color = COLORS[i]

        # plot, = plt.plot(time_step / xscale, max_performance, color=color)
        plot, = plt.plot(time_step / xscale, avg_performance, color=color)
        # print(algorithm_name + " : " + str(avg_performance[-1]) + " +- " + str(std_performance[-1]))
        # for j in range(total_iter):
        #     print(str(j) + " : " + str(arr_performance[j][-1]))
        arr_plot.append(plot)
        if with_std:
            upper_performance = avg_performance + std_performance
            lower_performance = avg_performance - std_performance
            plt.fill_between(time_step / xscale, lower_performance, upper_performance, color=color, alpha=0.2, linestyle='None')


    plt.xlim(0, max_timesteps/xscale)
    # plt.ylim(-130, 100)
    if with_title:
        plt.title(env_name)

    plt.xlabel("Time Steps (1e6)")
    plt.ylabel("Average Episode Reward Sum")
    plt.tight_layout()
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

    if arr_name_plot is not None:
        plt.legend(list(reversed(arr_plot)), list(reversed(arr_name_plot)))
    else:
        plt.legend(list(reversed(arr_plot)), list(reversed(arr_algorithm_name)))
    if not os.path.isdir('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures'):
        os.mkdir('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures')
    if not os.path.isdir('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures/'+filename):
        os.mkdir('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures/'+filename)
    if save_fig:
        if with_std:
            plt.savefig(os.path.join('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures/', filename, env_name_ + filename + ".pdf"))
        else:
            plt.savefig(os.path.join('/home/wisrl/syhan/DISC_related/log_ppo/Mujoco_figures/', filename, env_name_ + filename + "." + save_format))
    # plt.show()



# def plot_learning_curves(base_dir, env_name, arr_algorithm_name, num_actors, total_iter,
#                          max_timesteps=1e6, with_std=False, with_max_performance=False, arr_name_plot=None, with_title=True, with_all_actors=False,
#                          fig_size = (6, 4.5), save_filename='Performance', save_fig=True, save_format='pdf',xscale=1e6, min_step = 4000):
#     filename = save_filename
#     if with_max_performance:
#         filename = "Max_" + filename
#
#     time_step = (np.arange(0, int(max_timesteps / min_step) + 1) + 1) * min_step
#     plt.figure(figsize=fig_size)
#     arr_plot = []
#     for i, algorithm_name in enumerate(arr_algorithm_name):
#         # filename = filename + '_' + algorithm_name
#         arr_performance = []
#         arr_max_performance = []
#
#         arr_performance_t = []
#         arr_max_performance_t = []
#         arr_timestep_t = []
#         for iter in range(total_iter):
#             dir = os.path.join(base_dir, env_name, algorithm_name, 'iter'+str(iter+1), 'progress.csv')
#             result = load_progress(dir)
#
#             if 'total-samples' in result:
#                 time_step = np.array(result['total-samples']) / xscale * num_actors[i]
#             else:
#                 time_step = np.array(result['total-samples/0'])/xscale * num_actors[i]
#             arr_timestep_t.append(time_step)
#
#             if num_actors[i] == 1:
#                 if 'return-average' in result:
#                     arr_performance_t.append(rolling_window(result['return-average']))
#                     arr_max_performance_t.append(rolling_window(result['return-average']))
#                 else:
#                     arr_performance_t.append(rolling_window(result['return-average/0']))
#                     arr_max_performance_t.append(rolling_window(result['return-average/0']))
#             elif num_actors[i] > 1:
#                 arr_performance_actors = []
#                 for actor_num in range(num_actors[i]):
#                     arr_performance_actors.append(result['return-average/{i}'.format(i=actor_num)])
#
#                 arr_performance_t.append(rolling_window(np.mean(arr_performance_actors, axis=0)))
#                 arr_max_performance_t.append(rolling_window(np.max(arr_performance_actors, axis=0)))
#             else:
#                 return
#
#         min_len = np.min([len(arr_timestep_t[i]) for i in range(total_iter)])
#         time_step = arr_timestep_t[0][:min_len]
#         for iter in range(total_iter):
#             arr_performance.append(arr_performance_t[iter][:min_len])
#             arr_max_performance.append(arr_max_performance_t[iter][:min_len])
#
#         arr_performance = np.array(arr_performance)
#         # print(arr_performance.shape)
#         if with_max_performance:
#             avg_performance = np.mean(arr_max_performance, axis=0)
#             std_performance = np.std(arr_max_performance, axis=0)
#             max_performance = np.max(arr_max_performance, axis=0)
#         else:
#             avg_performance = np.mean(arr_performance, axis=0)
#             std_performance = np.std(arr_performance, axis=0)
#
#         color = COLORS[i]
#
#         # plot, = plt.plot(time_step, max_performance, color=color)
#         plot, = plt.plot(time_step, avg_performance, color=color)
#         print(algorithm_name + " : " + str(avg_performance[-1]) + " +- " + str(std_performance[-1]))
#         for j in range(total_iter):
#             print(str(j) + " : " + str(arr_performance[j][-1]))
#         arr_plot.append(plot)
#         if with_std:
#             upper_performance = avg_performance + std_performance
#             lower_performance = avg_performance - std_performance
#             plt.fill_between(time_step, lower_performance, upper_performance, color=color, alpha=0.2, linestyle='None')
#
#
#     plt.xlim(0, max_timesteps/xscale)
#     # plt.ylim(-130, 100)
#     if with_title:
#         plt.title(env_name)
#
#     plt.xlabel("Environment Steps (1e6)")
#     plt.ylabel("Average Rewards")
#     plt.tight_layout()
#     # plt.ylim(-20, 0)
#     plt.grid(True)
#
#     if arr_name_plot is not None:
#         plt.legend(list(reversed(arr_plot)), list(reversed(arr_name_plot)))
#     else:
#         plt.legend(list(reversed(arr_plot)), list(reversed(arr_algorithm_name)))
#
#     if save_fig:
#         if with_std:
#             plt.savefig(os.path.join(base_dir, 'figures_new', filename + "_" + env_name + "_fill.pdf"))
#         else:
#             plt.savefig(os.path.join(base_dir, 'figures_new', filename + "_" + env_name + "." + save_format))
#     plt.show()






# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files

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
    {'env_name': 'HalfCheetah-v1', 'max_timesteps': int(3e6)},                  #0
    {'env_name': 'Hopper-v1', 'max_timesteps': int(1e6)},             #3
    {'env_name': 'Walker2d-v1', 'max_timesteps': int(1e6)},            #4
    {'env_name': 'Ant-v1', 'max_timesteps': int(3e6)},#7
    {'env_name': 'Humanoid-v1', 'max_timesteps': int(1e7)},#8                 #1
    {'env_name': 'HumanoidStandup-v1', 'max_timesteps': int(1e7)},  # 2
    {'env_name': 'BipedalWalker-v2', 'max_timesteps': int(3e6)},                 #5
    {'env_name': 'BipedalWalkerHardcore-v2', 'max_timesteps': int(3e6)},  # 6
    {'env_name': 'Swimmer-v1', 'max_timesteps': int(1e6)},  # 2
    {'env_name': 'InvertedPendulum-v1', 'max_timesteps': int(1e6)},  # 6
    {'env_name': 'InvertedDoublePendulum-v1', 'max_timesteps': int(1e6)},  # 2
    {'env_name': 'Pendulum-v0', 'max_timesteps': int(1e6)},  # 2
    {'env_name': 'Reacher-v1', 'max_timesteps': int(1e6)},
]

ALGORITHM_INFO = [
    {'algorithm_name': 'ppo2_MBER_leng1_clip0.2_minlr0.0001', 'num_actors': 1, 'name_plot': 'PPO'},         #0
    {'algorithm_name': 'ppo2_AMBER_leng64_clip0.2_clipcut0.2_minlr0.0001', 'num_actors': 1, 'name_plot': 'PPO-AMBER($\epsilon_b=0.2$)'},         #1
    {'algorithm_name': 'ppo2_AMBER_leng64_clip0.2_vtr_rgae_clipcut0.2_minlr0.0001', 'num_actors': 1, 'name_plot': 'amber(v,cl0.2,cut0.2)'},         #2
    {'algorithm_name': 'ppo2_MBER_leng1_clip0.2_gradclip0.5_constlr_minsig0.2', 'num_actors': 1, 'name_plot': 'PPO(L=1,grad0.5,const)'},         #3
    {'algorithm_name': 'ppo2_MBER_leng1_clip0.2_vtr4_tr1.0_useadv_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=1,vtr,adv)'},         #4
    {'algorithm_name': 'ppo2_MBER_leng2_clip0.2_vtr4_tr1.0_useadv_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=2,vtr,adv)'},         #5
    {'algorithm_name': 'ppo2_MBER_leng4_clip0.2_vtr4_tr1.0_useadv_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=4,vtr,adv)'},         #6
    {'algorithm_name': 'ppo2_MBER_leng8_clip0.2_vtr4_tr1.0_useadv_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=8,vtr,adv)'},         #7
    {'algorithm_name': 'ppo2_MBER_leng1_clip0.2_vtr4_tr1.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=1,vtr1.0,gae)'},  # 8
    {'algorithm_name': 'ppo2_MBER_leng2_clip0.2_vtr4_tr1.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=2,vtr1.0,gae)'},  # 9
    {'algorithm_name': 'ppo2_MBER_leng4_clip0.2_vtr4_tr1.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=4,vtr1.0,gae)'},  # 10
    {'algorithm_name': 'ppo2_MBER_leng8_clip0.2_vtr4_tr1.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=8,vtr1.0,gae)'},  # 11
    {'algorithm_name': 'ppo2_MBER_leng1_clip0.2_vtr4_tr5.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=1,vtr5.0,gae)'}, # 12
    {'algorithm_name': 'ppo2_MBER_leng2_clip0.2_vtr4_tr5.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=2,vtr5.0,gae)'}, # 13
    {'algorithm_name': 'ppo2_MBER_leng4_clip0.2_vtr4_tr5.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=4,vtr5.0,gae)'}, # 14
    {'algorithm_name': 'ppo2_MBER_leng8_clip0.2_vtr4_tr5.0_rgae', 'num_actors': 1, 'name_plot': 'PPO(L=8,vtr5.0,gae)'}, # 15
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng1_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=1,vtr1.0,dtarg0.005)'}, # 16
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng2_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=2,vtr1.0,dtarg0.005)'},  # 17
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng4_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=4,vtr1.0,dtarg0.005)'},  # 18
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng8_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=8,vtr1.0,dtarg0.005)'},  # 19
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng1_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.01_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=1,vtr1.0,dtarg0.01)'},  # 20
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng2_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.01_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=2,vtr1.0,dtarg0.01)'},  # 21
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng4_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.01_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=4,vtr1.0,dtarg0.01)'},  # 22
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng8_clip0.1_vtr4_tr1.0_adap_kl_dtarg0.01_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=8,vtr1.0,dtarg0.01)'},  # 23
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng1_clip0.1_vtr4_tr5.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=1,vtr5.0,dtarg0.005)'},  # 24
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng2_clip0.1_vtr4_tr5.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=2,vtr5.0,dtarg0.005)'},  # 25
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng4_clip0.1_vtr4_tr5.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=4,vtr5.0,dtarg0.005)'},  # 26
    {'algorithm_name': 'ppo2_MBER4_clipdim2_leng8_clip0.1_vtr4_tr5.0_adap_kl_dtarg0.005_rgae', 'num_actors': 1, 'name_plot': 'PPO_clipdim(L=8,vtr5.0,dtarg0.005)'},  # 27
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr4_tr1.0_adap_kl_dtarg0.002_rgae_clipcut0.2', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v4,cl0.2,dt0.002,cut0.2)'},  # 28
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr4_tr1.0_adap_kl_dtarg0.005_rgae_clipcut0.2', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v4,cl0.2,dt0.005,cut0.2)'},  # 29
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr4_tr1.0_adap_kl_dtarg0.002_rgae_clipcut0.3', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v4,cl0.2,dt0.002,cut0.3)'},  # 30
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr4_tr1.0_adap_kl_dtarg0.005_rgae_clipcut0.3', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v4,cl0.2,dt0.005,cut0.3)'},  # 31
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.1', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.002,cut0.1)'},  # 32
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.1', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.005,cut0.1)'},  # 33
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.2', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.002,cut0.2)'},  # 34
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.2', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.005,cut0.2)'},  # 35
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.3', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.002,cut0.3)'},  # 36
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.3', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.005,cut0.3)'},  # 37
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.4', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.002,cut0.4)'},  # 38
    {'algorithm_name': 'ppo2_AMBER4_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.4', 'num_actors': 1, 'name_plot': 'AMBER_cdim(v,cl0.2,dt0.005,cut0.4)'},  # 39
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_clipcut0.06', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.001,cut0.06)'},  # 40
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.06', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.002,cut0.06)'},  # 41
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.06', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.005,cut0.06)'},  # 42
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_clipcut0.07', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.001,cut0.07)'},  # 43
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.07', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.002,cut0.07)'},  # 44
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.07', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.005,cut0.07)'},  # 45
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_clipcut0.08', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.001,cut0.08)'},  # 46
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.08', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.002,cut0.08)'},  # 47
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.08', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.005,cut0.08)'},  # 48
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_clipcut0.09', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.001,cut0.09)'},  # 49
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.002_rgae_clipcut0.09', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.002,cut0.09)'},  # 50
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.005_rgae_clipcut0.09', 'num_actors': 1, 'name_plot': 'AMBER5_cdim(v,cl0.2,dt0.005,cut0.09)'},  # 51


    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.001_rgae_constlr_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.001,cut0.09, const)'},  # 52
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.002_rgae_constlr_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.002,cut0.09, const)'},  # 53
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.005_rgae_constlr_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.005,cut0.09, const)'},  # 54
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.001_rgae_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.001,cut0.09)'},  # 55
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.002_rgae_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.002,cut0.09)'},  # 56
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.3_vtr_adap_kl_dtarg0.005_rgae_clipcut0.09_minsig0.2', 'num_actors': 1,
     'name_plot': 'AMBER5_cdim(cl0.3,dt0.005,cut0.09)'},  # 57


    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.1_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon = 0.1$'},  # 58
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon = 0.2$'},  # 59
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': 'DISC'},  # 60
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.8_vtr_adap_kl_dtarg0.001_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon = 0.8$'},  # 61
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.002_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': '$J_{IS}, J_{targ} = 0.002$'},  # 62
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.005_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
     'name_plot': '$J_{IS}, J_{targ} = 0.005$'},  # 63
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng1_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.05_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon_b = 0$'},  # 64
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.05_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon_b = 0.05$'},  # 65
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.2_minlr0.0001', 'num_actors': 1,
     'name_plot': '$\epsilon_b = 0.2$'},  # 66
    {'algorithm_name': 'ddpg', 'num_actors': 1, 'name_plot': 'DDPG'},  # 67
    {'algorithm_name': 'trpo_mpi', 'num_actors': 1, 'name_plot': 'TRPO'},  # 68
    {'algorithm_name': 'acktr', 'num_actors': 1, 'name_plot': 'ACKTR'},  # 69
    {'algorithm_name': 'trust-pcl', 'num_actors': 1, 'name_plot': 'Trust-PCL (off)'},  # 70

    {'algorithm_name': 'SQL', 'num_actors': 1, 'name_plot': 'SQL'},  # 71
    {'algorithm_name': 'test_TD3_ver2_NQ2', 'num_actors': 1, 'name_plot': 'TD3'}, # 72
    {'algorithm_name': 'SAC', 'num_actors': 1, 'name_plot': 'SAC'},  # 73
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_adap_kl_dtarg0.001_clipcut0.1_minlr0.0001', 'num_actors': 1,
         'name_plot': 'GAE'},  # 74

    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.1_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon = 0.1$'},  # 75
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.2_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon = 0.2$'},  # 76
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': 'DISC'},  # 77
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.8_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon = 0.8$'},  # 78
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.002_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$J_{KL}, J_{targ} = 0.002$'},  # 79
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.005_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$J_{KL}, J_{targ} = 0.005$'},  # 80
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng1_clip0.4_vtr_adap_kl_dtarg0.001_rgae_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon_b = 0$'},  # 81
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.05_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon_b = 0.05$'},  # 82
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_vtr_adap_kl_dtarg0.001_rgae_clipcut0.2_minlr0.0001',
     'num_actors': 1,
     'name_plot': '$\epsilon_b = 0.2$'},  # 83
    {'algorithm_name': 'ppo2_AMBER5_clipdim2_kloldnew_leng64_clip0.4_adap_kl_dtarg0.001_minlr0.0001', 'num_actors': 1,
         'name_plot': 'GAE'},  # 84

    {'algorithm_name': 'ppo2_AMBER5_clipdim2_leng64_clip0.4_vtr_rgae_clipcut0.1_minlr0.0001', 'num_actors': 1,
         'name_plot': r'no IS loss ($\alpha=0$)'},  # 85
    # {'algorithm_name': 'ppo2_AMBER_leng64_clip0.4_clipcut0.2_minlr0.0001', 'num_actors': 1, 'name_plot': 'PPO-AMBER($\epsilon=0.4$)'},         #1
    {'algorithm_name': 'ppo2_AMBER_leng64_clip0.2_clipcut0.1_minlr0.0001', 'num_actors': 1, 'name_plot': 'PPO-AMBER($\epsilon_b=0.1$)'},         #1
]
# {'algorithm_name': 'MPE_PPO_update5_TRatio20', 'num_actors': 4, 'name_plot': 'MPE_PPO_20'},
# {'algorithm_name': 'MPE_SAC_NA4_NQ2_update1_TRatio2_ver3', 'num_actors': 4, 'name_plot': 'IPE-SAC'},    #1


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

def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_ind', type=int, default=0)
    parser.add_argument('--base_dir', help = 'Base directories', default='/home/wisrl/syhan/DISC_related/log_ppo/Mujoco/')
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[52,53,54,55,56,57,'amber5_clip0.3'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[1,86,0,60,'_base'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[58,59,60,61,'_clip'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[85,60,62,63,77,79,80,'_IStarg'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[64,65,60,66,'_batchlim'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[74,60,'_nov'])
    parser.add_argument('--algorithm_ind', help='List of algorithms', default=[67,68,69,70,71,72,73,60, '_other'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[0,1,2,'_amber'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[0,1,77,'_klbase'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[75,76,77,78,'_klclip'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[77,79,80,'_klIStarg'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[81,82,77,83,'_klbatchlim'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[84,77,'_klnov'])
    # parser.add_argument('--algorithm_ind', help='List of algorithms', default=[0,67,68,69,70,71,72,73,77, '_klother'])
    parser.add_argument('--total-iter', type=int, default=5)
    parser.add_argument('--fig-size', help='Size of Figure (width, height)', default=(6,4.5))
    parser.add_argument('--save-fig', type=bool, default=True)
    parser.add_argument('--with-std', type=bool, default=True)
    parser.add_argument('--with-max-performance', type=bool, default=False)
    parser.add_argument('--with-title', type=bool, default=True)
    parser.add_argument('--with-all-actors', type=bool, default=False)
    parser.add_argument('--save-format', help='figure format (eps, fig, png, etc)', default="eps")

    args = parser.parse_args()
    for i in range(8):
        args.algorithm_name = [ALGORITHM_INFO[algorithm_ind]['algorithm_name'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.num_actors = [ALGORITHM_INFO[algorithm_ind]['num_actors'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.name_plot = [ALGORITHM_INFO[algorithm_ind]['name_plot'] for algorithm_ind in args.algorithm_ind[:-1]]
        args.env_name = ENV_INFO[i]['env_name']
        args.max_timesteps = ENV_INFO[i]['max_timesteps']

        plot_learning_curves(args.base_dir, args.env_name, args.algorithm_name, args.num_actors, args.total_iter,
                             max_timesteps=args.max_timesteps, with_std=args.with_std, with_max_performance=args.with_max_performance, with_title=args.with_title, arr_name_plot=args.name_plot,
                             fig_size=args.fig_size, save_filename=args.algorithm_ind[-1], save_fig=args.save_fig, save_format=args.save_format)
        # plot_best_selected_rate(args.base_dir, args.env_name, args.algorithm_name[0], args.num_actors[0], args.total_iter,
        #                         max_timesteps=args.max_timesteps, with_title=args.with_title,
        #                         fig_size=(6, 4.5), save_filename='Selected_Ratio', save_fig=args.save_fig, save_format=args.save_format)
        # plot_learning_curves_all_actors(args.base_dir, args.env_name, args.algorithm_name, args.num_actors, args.total_iter,
    #                                 max_timesteps=args.max_timesteps, arr_name_plot=args.name_plot, with_title=args.with_title,
    #                                 fig_size=args.fig_size, save_filename='Performance', save_fig=args.save_fig, save_format=args.save_format)

if __name__ == '__main__':
    main()
