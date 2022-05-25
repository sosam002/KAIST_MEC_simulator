import collections

import gym
from gym.envs.registration import register
from gym.spaces import Box
import numpy as np

from servernode_w_appqueue import ServerNode as ANode
from servernode_w_totalqueue import ServerNode as TNode
from channels import Channel
from constants import GHZ, WIRED
from cost_functions_cores import get_local_power_cost


register(id='MEC-v1', entry_point=MEC_v1)

class MEC_v1(gym.Env):
    # def __init__(self, tq_nodes, aq_nodes, channels, link_infos, task_rate=1, use_beta=True, cost_type=1, max_episode_steps=5000):
    def __init__(self, args):
        super().__init__()
        self.reset_info = args
        self.initialize()


        # it depends on state design. Edit this and get_state() function.
        state = self.get_status()
        self.state_dim = len(state)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(state),))

        # it depends on action design. Edit this.
        self.action_dim = 0
        for client in self.clients.values():
            self.action_dim += len(client.get_applications())+1
        if self.use_beta:
            self.action_dim *=2
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.action_dim,))


    def render(self):
        raise NotImplementedError()


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
        self.use_beta = True
        self.applications=set()
        self.timestamp = 0

        self.nodes = dict()
        self.clients = dict()
        self.servers = dict()
        self.channels = dict()
        self.links = list()


        for node in self.reset_info['tq_nodes']:
            node['uuid']= self.add_tq_node(node['id'], node['num_cores'], node['single_clk'], node['movable'])

        for node in self.reset_info['aq_nodes']:
            node['uuid']= self.add_aq_node(node['id'], node['num_cores'], node['single_clk'], node['applications'], node['is_random_task_generating'], node['offload_type'], node['movable'])
            self.applications.update(node['applications'])

        for channel in self.reset_info['channels']:
            channel['uuid']= self.add_channel(channel['id'], channel['channel_type'], channel['pathloss'], channel['lf'], channel['sf_type'], channel['sf_factor'], channel['rate'], channel['op_freq'])

        for link in self.reset_info['link_infos']:
            client = self.nodes[link['client_id']]
            self.clients[client.get_uuid()] = client
            server = self.nodes[link['server_id']]
            self.servers[server.get_uuid()] = server

            self.add_link(client, server, self.channels[link['channel_id']])

    def reset(self):
        self.initialize()
        reset_state = self.get_status(scale=GHZ)
        return reset_state

    def get_status(self, scale=GHZ, involve_link=False):
        edge_state, cloud_state, link_state = list(), list(), list()
        # failed_to_offload, failed_to_generate = 0, 0
        for client in self.clients.values():
            temp_state = client.get_status(self.timestamp, scale=scale)
            edge_state += temp_state
            # failed_to_generate += sum(temp_state[24:32])

        state = edge_state
        if self.use_beta:
            for server in self.servers.values():
                temp_state = server.get_status(self.timestamp, scale=scale)
                cloud_state += temp_state
                # failed_to_offload +=sum(temp_state[24:32])
            for link in self.links:
                link_state.extend([self.clients[link[0]].sample_channel_rate(link[1]),self.servers[link[1]].sample_channel_rate(link[0])])

        if involve_link:
            state = edge_state + cloud_state + link_state
        else:
            state = edge_state + cloud_state

        return np.array(state)#, failed_to_offload, failed_to_generate

    # several clients, several servers not considered. (step, _step_alpha, _step_beta)
    # def step(self, action, cloud, use_beta=True, generate=True):
    def step(self, action):
        q0, failed_to_generate, q1 = self._step_generation()
        action_alpha, action_beta, usage_ratio = list(), list(), list()
        if self.use_beta:
            action_alpha = action.flatten()[:int(self.action_dim/2)].reshape(1,-1)
            action_beta = action.flatten()[int(self.action_dim/2):].reshape(1,-1)
        else:
            action_alpha = action

        used_edge_cpus, inter_state, q2 = self._step_alpha(action_alpha)
        # used_cloud_cpus, new_state, failed_to_offload, q3 = self._step_beta(action_beta, np.array(cloud).reshape(-1,len(cloud)))
        # used_cloud_cpus, new_state = self._step_beta(action_beta, np.array(cloud).reshape(-1,len(cloud)))
        used_cloud_cpus, new_state = self._step_beta(action_beta)
        # fail_cost = self.get_fail_cost(failed_to_offload, failed_to_generate)

        # cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, q3, failed_to_offload, failed_to_generate)
        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, q0, q2)
        self.timestamp += 1
        if self.timestamp == self.max_episode_steps:
            return new_state, -cost, True, {}
        # return new_state, cost, failed_to_offload+failed_to_generate
        return new_state, -cost, False, {}

    def _step_alpha(self, action):
        # initial_qlength= self.get_total_qlength()
        used_edge_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        if self.timestamp%1000==0:
            print("alpha", 1-sum(sum(action)))
        for client_id, alpha in list(zip(self.clients.keys(), action)):
            used_edge_cpus[client_id] = self.clients[client_id].do_tasks(alpha)

        # state, _, _ = self.get_status()
        state = self.get_status(scale=GHZ)
        # after_qlength = self.get_total_qlength()
        after_qlength = self.get_edge_qlength(scale=GHZ)

        return used_edge_cpus, state, after_qlength


    # def _step_beta(self, action, action_cloud):
    def _step_beta(self, action):
        used_txs = collections.defaultdict(list)
        tasks_to_be_offloaded = collections.defaultdict(dict)
        used_cloud_cpus = collections.defaultdict(float)
        action = action.flatten()[:-1].reshape(1,-1)
        if self.timestamp%1000==0:
            print("beta", 1-sum(sum(action)))
        # 모든 client 객체에 대해 각 client의 상위 node로 offload하기
        # 각 client는 하나의 상위 노드를 가지고 있다고 가정함......?
        for client, beta in list(zip(self.clients.values(), action)):
            higher_nodes = client.get_higher_node_ids()
            for higher_node in higher_nodes:
                # import pdb; pdb.set_trace()/
                used_tx, task_to_be_offloaded, failed = client.offload_tasks(beta, higher_node)
                used_txs[higher_node].append(used_tx)
                tasks_to_be_offloaded[higher_node].update(task_to_be_offloaded)

        # server_action = dict(zip(self.servers.keys(), action_cloud))
        for server_id, server in self.servers.items():
            # import pdb; pdb.set_trace()
            server.offloaded_tasks(tasks_to_be_offloaded[server_id], self.timestamp)
            used_cloud_cpus[server_id] = server.do_tasks()

        # after_qlength = self.get_total_qlength()
        # state, failed_to_offload, _ = self.get_status()
        state = self.get_status(scale=GHZ)
        # return used_cloud_cpus, state, failed_to_offload, after_qlength
        return used_cloud_cpus, state

    def _step_generation(self):
        # initial_qlength= self.get_total_qlength()
        initial_qlength= self.get_edge_qlength()
        if not self.silence:
            print("###### random task generation start! ######")
        for client in self.clients.values():
            arrival_size, failed_to_generate = client.random_task_generation(self.task_rate, self.timestamp, *self.applications)
        if not self.silence:
            print("###### random task generation ends! ######")
        # after_qlength = self.get_total_qlength()
        after_qlength = self.get_edge_qlength(scale=GHZ)

        return initial_qlength, failed_to_generate, after_qlength


    def get_edge_qlength(self, scale=1):
        qlengths = list()
        for node in self.clients.values():
            for _, queue in node.get_queue_list():
                qlengths.append( queue.get_length(scale) )
        return qlengths

    def get_cloud_qlength(self, scale=1):
        qlengths = list()
        for node in self.servers.values():
            qlengths.append(node.get_task_queue_length(scale))
        return np.array(qlengths)


    def get_cost_old(self, used_edge_cpus, used_cloud_cpus, before, after, failed_to_offload=0, failed_to_generate=0):
        # drift_cost = get_drift_cost2(np.array(before),np.array(after),self.empty_reward)
        # import pdb; pdb.set_trace()

        lyap_drift_cost = sum(np.array(after)**2)-sum(np.array(before)**2)
        if lyap_drift_cost >0:
            lyap_drift_cost=np.sqrt(lyap_drift_cost)
        else:
            lyap_drift_cost=-np.sqrt(-lyap_drift_cost)
        edge_drift_cost = np.sqrt(sum(np.array(after)**2))
        #
        # after = np.array(after)
        # before = np.array(before)
        # condition = after-before
        # drift_cost = [sum((after+ (condition>0)*before)**2)]*100

        # fail_cost = get_fail_cost(failed_to_offload, failed_to_generate)*drift_cost[0]
        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            # local_cost += 10*(used_edge_cpu/10)**3
            edge_computation_cost += get_local_power_cost(used_edge_cpu)
        edge_computation_cost /= (900000*64*GHZ**3)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            cloud_payment_cost += 54*(used_cloud_cpu/54)**3
        cloud_payment_cost /= (900000*64*GHZ**3)
        # import pdb; pdb.set_trace()
        cloud_payment_cost += sum(self.get_cloud_qlength(scale=GHZ)**2)

        # import pdb; pdb.set_trace()

        if self.cost_type==1:
            return edge_drift_cost + edge_computation_cost + cloud_payment_cost
        elif self.cost_type==2:
            return edge_drift_cost + edge_computation_cost + 10*cloud_payment_cost
        elif self.cost_type==3:
            return edge_drift_cost + 1000*edge_computation_cost + 10*1000*cloud_payment_cost
        elif self.cost_type==4:
            return edge_drift_cost + 10*1000*edge_computation_cost + 1000*cloud_payment_cost
        elif self.cost_type==5:
            return edge_drift_cost + 100*1000*edge_computation_cost + 1000*cloud_payment_cost
        else:
            return lyap_drift_cost + edge_computation_cost + cloud_payment_cost

        # return total_cost(used_edge_cpus, used_cloud_cpus, drift_cost, option=self.cost_type)# + fail_cost
        # return drift_cost[0]
    def get_cost(self, used_edge_cpus, used_cloud_cpus, before, after, failed_to_offload=0, failed_to_generate=0):

        # import pdb; pdb.set_trace()
        # for 3 apps
        edge_drift_cost = sum(np.array(after)**2)
        # for 8 apps
        # edge_drift_cost = 10*sum(np.array(after)**2)

        edge_computation_cost = 0
        for used_edge_cpu in used_edge_cpus.values():
            if self.cost_type==10 or self.cost_type==11 or self.cost_type==14 or self.cost_type==16 or self.cost_type>99:
                edge_computation_cost += 10*(used_edge_cpu/10)**3
            else: # 12, 13, 15
                edge_computation_cost += get_local_power_cost(used_edge_cpu)
        edge_computation_cost /= (900000*64*GHZ**3)

        cloud_payment_cost = 0
        for used_cloud_cpu in used_cloud_cpus.values():
            # for 3 apps
            cloud_payment_cost += 54*(used_cloud_cpu/54)**3
            # for 8 apps
            # cloud_payment_cost += 45*(used_cloud_cpu/45)**3
        cloud_payment_cost /= (900000*64*GHZ**3)


        if self.cost_type==10 or self.cost_type==12:
            return 100*edge_drift_cost + 1*edge_computation_cost + 50*cloud_payment_cost
        elif self.cost_type==11 or self.cost_type==13:
            return 100*edge_drift_cost + 50*edge_computation_cost + 1*cloud_payment_cost
        elif self.cost_type==14 or self.cost_type==15:
            return 100*edge_drift_cost + 100*edge_computation_cost + 1*cloud_payment_cost
        elif self.cost_type==16 or self.cost_type==17:
            return 100*edge_drift_cost + 100*edge_computation_cost + 0.1*cloud_payment_cost
        elif self.cost_type>99:
            return self.cost_type*edge_drift_cost/100 + 10*edge_computation_cost + 10*cloud_payment_cost
        else:
            return (10**self.cost_type)*edge_drift_cost/100 + 10*edge_computation_cost + 10*cloud_payment_cost
