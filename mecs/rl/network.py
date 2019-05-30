import tensorflow as tf
import numpy as np
import random
from collections import deque

class resourceAlloc:

    REPLAY_MEMORY = 100000
    BATCH_SIZE = 32
    GAMMA = 0.8
    STATE_LEN = 100

    def __init__(self, session, state_size, action_size):
        self.session = session

        # queue length of 8 applications, avg arrival for every queue (8+8), and.. what?
        self.state_size = state_size
        # alpha1,...,alphaN, beta1,...,betaN --> 두개 네트워크를 따로 만들어야..?아니이게무슨!!ㅠㅠ
        self.action_size = action_size

        self.memory = deque()

        self.learning_rate = 1e-4
        self.hidden_size = [512, 256]
        # input state
        self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
        # input action
        self.action = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action')
        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target')

        self.Q = self.build_network('main')
        self.target_Q = self.build_network('target')



    def build_network(self, name, duel=True):
        with tf.variable_scope(name):
            '''
            뉴럴넷 구성, 뭐로 하나.. conv? 그냥? 그냥이 좋겟다..
            '''
            W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size[0]], stddev=0.01), name="W1")
            b1 = tf.Variable(tf.random_normal([self.hidden_size[0]], stddev=0.01), name="b1")
            L1 = tf.nn.relu(tf.matmul(self.X, W1)+b1)

            W2 = tf.Variable(tf.random_normal([self.hidden_size[0], self.hidden_size[1]], stddev=0.01), name="W2")
            b2 = tf.Variable(tf.random_normal([self.hidden_size[1]], stddev=0.01), name="b2")
            L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

            W3 = tf.Variable(tf.random_normal([self.hidden_size[1], self.output_size], stddev=0.01), name="W3")
            b3 = tf.Variable(tf.random_normal([self.output_size], stddev=0.01), name="b3")


            model = self.state
            action_values = tf.layers.dense(model, self.n_action, activation=None)

            if duel == True:
                v_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], 1], stddev=0.01), name="v_W3")
                v_b3 = tf.Variable(tf.random_normal([1], stddev=0.01), name="v_b3")
                weights['v_W3']=v_W3
                weights['v_b3']=v_b3
                v = tf.matmul(L2, v_W3)+v_b3
                advantage = tf.matmul(L2, W3)+b3
                actions_q = v + (advantage - tf.reduce_mean(advantage, reduction_indices = 1, keepdims = True))
            else:
                actions_q = tf.matmul(L2, W3)+b3

        return actions_q
        # self.predictions = tf.nn.softmax(self.action_values)

    #이건 discrete한..
    def get_action(self):
        actions_q = tf.nn.softmax(self.Q)
        actions_q = self.session.run(actions_q, feed_dict = {self.state: [self.state]})
        return actions_q

    def local_cost(self, used_cpu, used_tx):
        cost =  tf.add(tf.square(used_cpu), tf.square(used_tx))
        return cost
        # energy 소비 + 가격 cost식
    def offload_cost(self, used_tx, workloads):
        cost = np.array(used_tx)*np.array(workloads)
        return cost

    def queue_length_penalty(self, queue_lengths, max_queue_lengths):
        # queue length에 따라 어떻게 - reward 줄 지.
        remaind_queues = np.square(max_queue_lengths-queue_lengths)
        return np.sum(remained_queues)

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def sars_memory(self, state, action, reward):
        next_state = np.reshape(state, (self.state_size, 1))
        next_state = np.append(self.state[:, 1:], next_state, axis=1)

        self.memory.append((self.state, next_state, action, reward))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state


    def update():
        pass
