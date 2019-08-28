class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def evaluate_policy(env, policy, cloud_policy, memory, epsd_length=1000, eval_episodes=10):
    print("---------------------------------------")
    print("EVALUATION STARTED")
    print("---------------------------------------")
    eval_mem = Memory()
    # import pdb; pdb.set_trace()
    eval_mem.actions = memory.actions[-epsd_length:]
    eval_mem.states = memory.states[-epsd_length:]
    eval_mem.logprobs = memory.logprobs[-epsd_length:]
    eval_mem.rewards = memory.rewards[-epsd_length:]
    avg_reward = 0.
    for _ in range(eval_episodes):
    	obs = env.reset()
    	done = False
    	costs=[]

    	for t in range(epsd_length):
    		silence=True
    		if t%200==0:
    			silence=False

    		action = policy.select_action(obs, eval_mem)
    		# import pdb; pdb.set_trace()
    		obs, cost, failed = env.step_together(t, np.array(action).reshape(-1,len(action)), cloud_policy, silence=silence)

    		avg_reward -= cost
    		costs.append(cost)

    		if failed or t==999:
    			print("episode length {}".format(t))
    			break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward
