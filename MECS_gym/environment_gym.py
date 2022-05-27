import collections

import gym
from gym.envs.registration import register
from gym.spaces import Box
import numpy as np

from servernode_w_appqueue import ServerNode as ANode
from servernode_w_totalqueue import ServerNode as TNode
from channels import Channel
from constants import *
from applications import *


class MEC(gym.Env):
    def __init__(self, args):
        super().__init__()
        self.reset_info = args
        self.initialize()


        # it depends on state design. Edit this and get_state() function.
        state = self._get_obs()
        self.state_dim = len(state)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(state),))

        # it depends on action design. Edit this.
        self.action_dim = 0
        for client in self.clients.values():
            self.action_dim += len(client.get_applications())+1
        if self.use_beta:
            self.action_dim *=2
        self.action_space = Box(low=-2, high=2, shape=(self.action_dim,))


    def add_link(self, client, server, up_channel, down_channel=None):
        if not down_channel:
            down_channel = up_channel
        else:
            down_channel = down_channel

        client.links_to_higher[server.get_uuid()]= {
            'node' : server,
            'channel' : up_channel
        }
        server.links_to_lower[client.get_uuid()] = {
            'node' : client,
            'channel' : down_channel
        }
        self.links.append((client.get_uuid(), server.get_uuid()))
        return

    def add_aq_node(self, id, num_cores, single_clk, applications, is_random_task_generating=True, offload_type=1, movable=False):
        node = ANode(num_cores, single_clk, applications, is_random_task_generating, offload_type, movable)
        self.nodes[id] = node
        return node.get_uuid()
        # return client

    def add_tq_node(self, id, num_cores, single_clk, movable=False):
        node = TNode(num_cores, single_clk, movable)
        self.nodes[id] = node
        return node.get_uuid()
        # return server

    def add_channel(self, id, channel_type=WIRED, pathloss=False, lf=False, sf_type=False, sf_factor=0, rate=None,
                    op_freq=None):
        channel = Channel(channel_type, pathloss, lf, sf_type, sf_factor, rate, op_freq)
        self.channels[id] = channel
        return channel.get_uuid()


    def get_number_of_apps(self):
        return len(self.applications)

    def initialize(self):
        self.task_rate = self.reset_info['task_rate'] #time_delta
        self.cost_type = self.reset_info['cost_type']
        self.max_episode_steps = self.reset_info['max_episode_steps']
        self.scale = self.reset_info['scale']
        self.use_beta = True
        self.applications=set()
        self.timestamp = 0

        self.nodes = dict()
        self.clients = dict()
        self.servers = dict()
        self.channels = dict()
        self.links = list()

        self.lambdas = list()

        for node in self.reset_info['tq_nodes']:
            node['uuid']= self.add_tq_node(node['id'], node['num_cores'], node['single_clk'], node['movable'])

        for node in self.reset_info['aq_nodes']:
            node['uuid']= self.add_aq_node(node['id'], node['num_cores'], node['single_clk'], tuple(node['applications']), node['is_random_task_generating'], node['offload_type'], node['movable'])
            for app in node['applications']:
                import pdb; pdb.set_trace()
                self.lambdas.append(arrival_bits(app, dist='deterministic') * get_info(app, "popularity"))
            self.applications.update(node['applications'])

        self.lambdas = np.array(self.lambdas) * self.task_rate / self.scale

        for channel in self.reset_info['channels']:
            channel['uuid']= self.add_channel(channel['id'], channel['channel_type'], channel['pathloss'], channel['lf'], channel['sf_type'], channel['sf_factor'], channel['rate'], channel['op_freq'])

        for link in self.reset_info['link_infos']:
            client = self.nodes[link['client_id']]
            self.clients[client.get_uuid()] = client
            server = self.nodes[link['server_id']]
            self.servers[server.get_uuid()] = server

            self.add_link(client, server, self.channels[link['channel_id']])

    def _reset(self):
        self.initialize()
        self.before_arrival = self.get_edge_qlength()
        _, failed_to_generate, _ = self._step_generation()
        reset_state = self._get_obs()
        reset_state[3:6] = self.before_arrival # q(t)
        reset_state = self._get_obs()
        reset_state[-1] = 0.0
        return reset_state

    def _get_obs(self, involve_link=False):
        edge_state, cloud_state, link_state = list(), list(), list()
        # failed_to_offload, failed_to_generate = 0, 0
        for client in self.clients.values():
            temp_state = client._get_obs(self.timestamp, scale=self.scale)
            edge_state += temp_state
            # failed_to_generate += sum(temp_state[24:32])

        state = edge_state
        if self.use_beta:
            for server in self.servers.values():
                temp_state = server._get_obs(self.timestamp, scale=self.scale)
                cloud_state += temp_state
                # failed_to_offload +=sum(temp_state[24:32])
            for link in self.links:
                link_state.extend([self.clients[link[0]].sample_channel_rate(link[1]),self.servers[link[1]].sample_channel_rate(link[0])])

        if involve_link:
            state = edge_state + cloud_state + link_state
        else:
            state = edge_state + cloud_state

        return np.array(state[:-3])#, failed_to_offload, failed_to_generate

    # several clients, several servers not considered. (step, _step_alpha, _step_beta)
    # def step(self, action, cloud, use_beta=True, generate=True):
    def step(self, action):
        # print(action)
        q0 = np.array(self.before_arrival) #q(t)
        action = 5 * action
        start_state = self._get_obs()
        q1=np.array(self.get_edge_qlength()) #q(t)+a(t)
        a_t = start_state[3:6]
        action_alpha, action_beta, usage_ratio = list(), list(), list()
        if self.use_beta:
            # action_alpha = np.array(list(action[:int(len(action)/2)]) + [0.0]).reshape(1,-1)
            # action_beta = np.array(list(action[int(len(action)/2):]) + [0.0]).reshape(1,-1)
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
            ### softmax here
            action_beta = softmax_1d(action_beta)
        else:
            action_alpha = action
        ### softmax here
        action_alpha = softmax_1d(action_alpha)
        # action_alpha = np.array([[0,0,0,1]])
        # action_beta = np.array([[0,0,0,1]])

        if self.timestamp%1000==0:
            print("--------------------------------------------")
            print("action", action)
            print("action_alpha", action_alpha)
            print("action_beta", action_beta)
            print("q_length",q1)


        # print("--------------------------------------------")
        # print("action", action)
        # print("action_alpha", action_alpha)
        # print("action_beta", action_beta)
        # print(used_edge_cpus)
        # print(used_cloud_cpus)
        used_edge_cpus, inter_state = self._step_alpha(action_alpha)
        ## print("action_beta", action_beta)
        # if self.timestamp < 500000:
        #     cost_type = 1000
        # else:
        #     cost_type = np.copy(self.cost_type)

        used_cloud_cpus, new_state, q_last = self._step_beta(action_beta)
        self.before_arrival = q_last #q(t+1)
        q2 = np.array(q_last) #q(t+1)=q(t)+a(t)-b(t)
        at_bt=q2-q0
        bt = q2 - q1
        _, failed_to_generate, _ = self._step_generation()

        new_state= self._get_obs()
        new_state[3:6] = q2
        q3 = new_state[6:9] #q(t+1)+a(t+1)

        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, bt)
        new_state[-1] = list(used_cloud_cpus.values())[0]/self.scale/216
        self.timestamp += 1
        if self.timestamp == self.max_episode_steps:
            return new_state, -cost, 1, {"cloud_cpu_used":start_state[-1]}
        return new_state, -cost, 0, {"cloud_cpu_used":start_state[-1]}

    def _step_alpha(self, action):
        # initial_qlength= self.get_total_qlength()
        used_edge_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)
        state = self._get_obs()

        if self.timestamp%1000==0:
            print("alpha", 1-sum(sum(action)))
        return used_edge_cpus, state


    # def _step_beta(self, action, action_cloud):
    def _step_beta(self, action):
        used_txs = collections.defaultdict(list)
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        # 모든 client 객체에 대해 각 client의 상위 node로 offload하기
        # 각 client는 하나의 상위 노드를 가지고 있다고 가정함......?
        q_before = self.get_edge_qlength()
        # print(q_before)
        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node)
                used_txs[higher_node].append(used_tx)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)
        q_last = self.get_edge_qlength()
        # print(np.array(q_last)-np.array(q_before))
        # print((np.array(q_last)-np.array(q_before))*np.array([10000,20000,40000]))
        s1 = self._get_obs()
        for server_id, server in self.servers.items():
            # print("s1",server.task_queue.get_cpu_needs(scale=1))
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], self.timestamp)
            # print("s2",server.task_queue.get_cpu_needs(scale=1))
            s2 = self._get_obs()
            used_cloud_cpus[server_id] = server.do_tasks()
            used_cloud_cpus[server_id] = (s2-s1)[-2]*GHZ


        state = self._get_obs()
        if self.timestamp%1000==0:
            print("beta", 1-sum(sum(action)))
        return used_cloud_cpus, state, q_last

    def _step_generation(self):
        initial_qlength= self.get_edge_qlength()
        for client in self.clients.values():
            arrival_size, failed_to_generate = client.random_task_generation(self.task_rate, self.timestamp, self.applications)

        after_qlength = self.get_edge_qlength()

        return initial_qlength, failed_to_generate, after_qlength


    def get_edge_qlength(self):
        qlengths = list()
        for node in self.clients.values():
            for _, queue in node.get_queue_list():
                qlengths.append( queue.get_length(scale=self.scale) )
        return qlengths

    def get_cloud_qlength(self):
        qlengths = list()
        for node in self.servers.values():
            qlengths.append(node.get_task_queue_length(scale=self.scale))
        return np.array(qlengths)

    def get_cost(self, used_edge_cpus, used_cloud_cpus,before, bt, failed_to_offload=0, failed_to_generate=0):

        def compute_cost_fct(cores, cpu_usage):
            return cores*(cpu_usage/400/GHZ/cores)**3

        edge_drift_cost = sum( 2*before*(self.lambdas+bt)+(self.lambdas+bt)**2 )


        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            edge_computation_cost += compute_cost_fct(10,used_edge_cpu)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            cloud_payment_cost += compute_cost_fct(54,used_cloud_cpu)


        if self.timestamp%1000==0:
            print("used cpu edge : ", used_edge_cpus.values())
            print("used cpu cloud : ", used_cloud_cpus.values())
            print("edge power : ", edge_computation_cost)
            print("cloud power : ", cloud_payment_cost)
            print("power * cost : ", self.cost_type*(edge_computation_cost+cloud_payment_cost))
            print("cost : ", edge_drift_cost+self.cost_type*(edge_computation_cost+cloud_payment_cost))
            print("rew : ", -(edge_drift_cost+self.cost_type*(edge_computation_cost+cloud_payment_cost)))


        return (edge_drift_cost+self.cost_type*(edge_computation_cost+cloud_payment_cost))
