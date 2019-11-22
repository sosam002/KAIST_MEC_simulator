import copy
import numpy as np

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

def evaluate_policy_new_network(env, policy, cloud_policy, memory, epsd_length=1000, eval_episodes=10, empty_reward=True):
    print("---------------------------------------")
    print("EVALUATION STARTED")
    print("---------------------------------------")
    eval_mem = Memory()
    # import pdb; pdb.set_trace()
    eval_mem.actions = list(memory.actions)
    eval_mem.states = list(memory.states)
    eval_mem.logprobs = list(memory.logprobs)
    eval_mem.rewards = list(memory.rewards)
    avg_rewards = []
    for _ in range(eval_episodes):
        avg_reward = 0
        obs = env.reset(empty_reward)
        done = False
        silence=True
        for t in range(epsd_length):
            action = policy.select_action(obs, eval_mem)
            # import pdb; pdb.set_trace()
            obs, cost, failed = env.step(np.array(action).reshape(-1,len(action)), cloud_policy)

            avg_reward -= cost

            if failed or t==epsd_length-1:
                avg_rewards.append(avg_reward)
                print("episode length {}".format(t))
                break

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    del eval_mem
    return avg_rewards

def evaluate_policy(env, policy, cloud_policy, memory, epsd_length=1000, eval_episodes=10, empty_reward=True):
    print("---------------------------------------")
    print("EVALUATION STARTED")
    print("---------------------------------------")
    eval_mem = Memory()
    # import pdb; pdb.set_trace()
    eval_mem.actions = list(memory.actions)
    eval_mem.states = list(memory.states)
    eval_mem.logprobs = list(memory.logprobs)
    eval_mem.rewards = list(memory.rewards)
    avg_rewards = []
    for _ in range(eval_episodes):
        avg_reward = 0
        obs = env.reset(empty_reward)
        done = False
        silence=True
        for t in range(epsd_length):
            action = policy.select_action(obs, eval_mem)
            # import pdb; pdb.set_trace()
            obs, cost, failed = env.step(t, np.array(action).reshape(-1,len(action)), cloud_policy, silence=silence)

            avg_reward -= cost

            if failed or t==epsd_length-1:
                avg_rewards.append(avg_reward)
                print("episode length {}".format(t))
                break

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    del eval_mem
    return avg_rewards
