import json
import torch
import logging
import os
import shutil

# from mecs
import scheduler, config
# from wholemap import WholeMap

from servernode_w_queue import ServerNode

from applications import *
from channels import *
from utilities import *
from constants import *
import rl.td3 as TD3
import environment

logger = logging.getLogger(__name__)

def evaluate_policy(env, policy, cloud_policy, eval_episodes=2):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		number_of_apps = env.get_number_of_apps()
		for t in range(100):

			action = policy.select_action(np.array(obs))
			# obs, cost = env.step(np.array(action[:len(app_info)]).reshape(-1,len(app_info)), np.array(action[len(app_info):]).reshape(-1,len(app_info)), np.array(cloud_policy).reshape(-1,len(cloud_policy)), t)
			obs, cost = env.step(np.array(action).reshape(-1,len(action)), np.array(cloud_policy).reshape(-1,len(cloud_policy)), t)
			avg_reward -= cost
		print(policy.actor.l1.weight)


	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward

def main():
    # main args로 받아올 것들?
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--policy_name", default="TD3")					# Policy name
	# parser.add_argument("--env_name", default="HalfCheetah-v1")			# OpenAI gym environment name
	# parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	# parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	# parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	# parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	# parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	# parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	# parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	# parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	# parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	# parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	# parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	# parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	# args = parser.parse_args()

	seed = 0
	start_timesteps = 1e4
	max_episode_steps = 4000
	# eval_freq = 5e3
	eval_freq = 4*max_episode_steps
	max_timesteps = 50*max_episode_steps
	expl_noise = 0.2
	expl_dec = 0.01/max_timesteps
	batch_size = 100
	discount = 0.6
	tau = 0.005
	policy_noise = 0.2
	noise_clip = 0.5
	policy_freq = 20
	save_models = True
	use_beta = False

	# 원래 gym env에서 주는 것들?




	log_dir = 'result_sosam'
	mobile_log = {}
	if os.path.isdir(log_dir):
	    shutil.rmtree(log_dir)
	os.mkdir(log_dir)

	if not os.path.exists("./results"):
	    os.makedirs("./results")
	if save_models and not os.path.exists("./pytorch_models"):
	    os.makedirs("./pytorch_models")

	file_name = 'test'




    # 지금은 policy가 없으니까 내맘대로. edge_policy
    # alpha = [0.5,0.5]
    # beta = [0.5,0.5]


    # TD3....eeeeeee
	# env = gym.make(args.env_name)
    ###################
	edge_capability = 2.5*1e2*GHZ
	cloud_capability = 2.5*1e4*GHZ  # clock per tick
	channel = WIRED
	applications = SPEECH_RECOGNITION,NLP, FACE_RECOGNITION#, SEARCH_REQ, LANGUAGE_TRANSLATION, PROC_3D_GAME, VR, AR
	number_of_apps = len(applications)
	cloud_policy = [1/number_of_apps]*number_of_apps
    ###################
	env = environment.Environment_sosam(1, *applications, use_beta=use_beta)

	torch.manual_seed(seed)
	np.random.seed(seed)

# Set seeds
# env.seed(args.seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

	obs = env.init_for_sosam(edge_capability, cloud_capability, channel)
	state_dim = env.state_dim
	# state_dim = len(app_info)*3+3
	# action_dim = len(app_info)*2
	action_dim = env.action_dim
	max_action = 1
	policy = TD3.TD3(state_dim, action_dim, max_action, use_beta)
	replay_buffer = ReplayBuffer()
	# evaluations = [evaluate_policy(env, policy, cloud_policy)]
	evaluations = []

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True




	while total_timesteps < max_timesteps:

		if done:
			if total_timesteps != 0:
				print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
				# if total_timesteps>0:
				# 	obs = env.reset()
				policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
				# if args.policy_name == "TD3":
				    # policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
				# else:
				# 	policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

	        # Evaluate episode
			if timesteps_since_eval >= eval_freq:
				timesteps_since_eval %= eval_freq
				evaluations.append(evaluate_policy(env, policy, cloud_policy))


				if save_models: policy.save(file_name, directory="./pytorch_models")
				np.save("./results/%s" % (file_name), evaluations)

			# Reset environment

			if total_timesteps >0:
				obs = env.reset()
				# del env
				# env = environment.Environment_sosam(1, *applications, use_beta=use_beta)
				# obs = env.init_for_sosam(edge_capability, cloud_capability, channel)


			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1


		# Select action
		action = policy.select_action(np.array(obs))
		if total_timesteps%200==0:
			print("#############################action####{}".format(action))
	    # action = [0.5,0.5]
	    # 이거 노이즈 주는 방법 알아야 할 듯.. softmax로 하는데..
		if expl_noise !=0:
			# import pdb; pdb.set_trace()
			# expl_noise = expl_noise*(1-expl_dec*total_timesteps)
			noise = np.random.normal(0.2*(1-expl_dec*total_timesteps), expl_noise*(1-expl_dec*total_timesteps), size=action_dim)
			# noise = abs(noise)/action.clip(0.1,0.9)
			# action = (action+np.random.normal(0.2,noise,size=action_dim)).clip(0.01)
			action = action+noise
			# if total_timesteps%200==0:
				# import pdb; pdb.set_trace()
			# action = np.concatenate( (action[:len(app_info)] / np.sum(action[:len(app_info)]), action[len(app_info):] / np.sum(action[len(app_info):])) )
			if use_beta:
				action = np.concatenate( (action[:number_of_apps] / np.sum(action[:number_of_apps]), action[number_of_apps:] / np.sum(action[number_of_apps:])) )
			else:
				action = action/np.sum(action)
			# action = np.stack( (action[:len(app_info)] / np.sum(action[:len(app_info)]), action[len(app_info):] / np.sum(action[len(app_info):])) )
			if total_timesteps%200==0:
				print("#################################action####{}".format(action))

		#
		# if expl_noise != 0:
		# 	action = action + np.random.normal(0, expl_noise, size= action_dim))

		# 	action[:len(app_info)] = action[:len(app_info)] / np.sum(action[:len(app_info)])


		# new_obs, cost = env.step(np.array(action[:len(app_info)]).reshape(-1,len(app_info)), np.array(action[len(app_info):]).reshape(-1,len(app_info)), np.array(cloud_policy).reshape(-1,len(cloud_policy)), episode_timesteps)
		new_obs, cost = env.step(np.array(action).reshape(-1,len(action)), np.array(cloud_policy).reshape(-1,len(cloud_policy)), episode_timesteps)
        # new_obs, cost = env.step(alpha, beta, cloud_policy, total_timesteps)

		# done_bool = 0 if episode_timesteps + 1 == max_episode_steps else 1
		if episode_timesteps == max_episode_steps:
			done = True
		done_bool = 1-float(done)

		episode_reward -= cost

		replay_buffer.add((obs, new_obs, action, -cost, done_bool))
		if total_timesteps%200==0:
			print("state: {}".format(obs))
			print("new state: {}".format(new_obs))
			print("cost:{}, reward:{}, episode reward{}".format(cost, -cost, episode_reward))

		obs = new_obs
		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval +=1


        # if t % 1000 == 0:
        #     logger.info(
        #         "================= << [%d,%d] second >> =================",
        #         t // 1000, t // 1000 + 1)
        #     logger.debug(json.dumps(mobile_log))
        #     mobile_log = {}
        # mobile_log[t] = my_map.simulate_one_time(t)
    # Final evaluation
    # evaluations.append(evaluate_policy(policy))

	evaluations.append(evaluate_policy(env, policy, cloud_policy))
	import pdb; pdb.set_trace()
	if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results/%s" % (file_name), evaluations)

if __name__ == "__main__":
    config.initialize_mecs()
    main()
