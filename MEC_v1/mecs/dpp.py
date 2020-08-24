from scipy.optimize import minimize
from constants import *
import numpy as np


class KKT_actor(object):
    def __init__(self, f_E, f_C, B, cost_type, scale=1):
        self.f_E = f_E
        self.f_C = f_C
        self.B = B
        self.cost_type = cost_type
        self.scale = scale

    def optimize(self, state):
        # queue_estimated_arrivals[app_type-1] = queue.mean_arrival(time, estimate_interval, scale=GHZ)
        # queue_arrivals[app_type-1] = queue.last_arrival(time, scale=GHZ)
        # queue_lengths[app_type-1] = queue.get_length(scale=1)
        # cpu_used[app_type-1] = self.cpu_used[app_type]/self.computational_capability
        # app_info[app_type-1] = applications.get_info(app_type, "workload")/KB

        state1 = state[:-4].reshape(5,-1)
        arrivals = state1[1]*GHZ
        q_lengths = state1[2]*GHZ
        cpu_used = state1[3]*self.f_E
        app_info = state1[4]*KB
        state2 = state[-4:]
        c_arrivals = state2[1]*GHZ
        c_q_lengths = state2[2]
        c_cpu_used = state2[3]*self.f_C
        action = [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]

        def get_local_power_cost(used_cpu, used_tx=0):
            cores, remained = divmod(used_cpu, 4*GHZ)
            return cores*(4*GHZ)**3+(remained)**3

        def objective(action):
            edge_drift_cost = 0
            for i in range(3):
                edge_drift_cost += q_lengths[i]*( arrivals[i]-action[i]*self.f_E/app_info[i]-action[i+3]*self.B )/GHZ/GHZ*self.scale
                # edge_drift_cost += (q_lengths[i]+ arrivals[i]-action[i]*self.f_E/app_info[i]-action[i+3]*self.B)**2/GHZ/100
                # edge_drift_cost -= q_lengths[i]**2/GHZ/100
            # if self.cost_type==6 or self.cost_type==7 or self.cost_type==8 or self.cost_type==9:
            #     edge_drift_cost = np.sqrt(edge_drift_cost)
            #     self.drift_coeff==1

            edge_computation_cost = ( (sum( action[:3] )*self.f_E)**3/100 )/ (10*GHZ)**3*self.scale

            # cloud_cost = min((c_q_lengths+((action[3]*app_info[0]+action[4]*app_info[1]+action[5]*app_info[2])*self.B)),self.f_C)**3/(54**2)/(10*GHZ)**3
            # import pdb; pdb.set_trace()
            cloud_cost = (action[3]*app_info[0]+action[4]*app_info[1]+action[5]*app_info[2])*self.B**3/(54**2)/(10*GHZ)**3*self.scale
            # if c_q_lengths>0:
            #     import pdb; pdb.set_trace()
            return edge_drift_cost + self.cost_type*(edge_computation_cost + cloud_cost)

        # constraints
        consts = []
        def constraint1(x):
            return (1 - sum(x[:3]))*1e2
        consts += [{'type':'ineq', 'fun':constraint1}]

        def constraint2(x):
            return (1 - sum(x[3:]))*1e2
        consts += [{'type':'ineq', 'fun':constraint2}]

        def const_function(k):
            # return lambda action: q_lengths[k] + arrivals[k] - (action[k]*self.f_E/app_info[k]+action[k+3]*self.B)
            def constraint(x):
                return x[k]*1e2
            return constraint
            # return lambda action: action[k]
        for i in range(6):
            consts += [{'type':'ineq', 'fun':const_function(i)}]

        def const_function2(k):
            # return lambda action: q_lengths[k] + arrivals[k] - (action[k]*self.f_E/app_info[k]+action[k+3]*self.B)
            # return lambda action: q_lengths[k] + arrivals[k] - (action[k]*self.f_E/app_info[k]+action[k+3]*self.B)
            def constraint(x):
                # return (q_lengths[k] + arrivals[k] - (x[k]*self.f_E/app_info[k]+x[k+3]*self.B))*1e2-0.000000001
                return (q_lengths[k] + arrivals[k] - (x[k]*self.f_E/app_info[k]+x[k+3]*self.B))
            return constraint
        for i in range(3):
            consts +=  [{'type':'ineq', 'fun':const_function2(i)}]

        #
        # for i in range(6):
        #     consts.append({'type':'ineq', 'fun':constraints[i]})
        # for i in range(3):
        #     consts.append({'type':'ineq', 'fun':constraints2[i]})

        cons= tuple(consts)


        # cons = tuple([const1]+[const2]+consts)
        # import pdb; pdb.set_trace()
        # bounds
        b = (0.0, 1.0)
        bnds = [b]*6

        # solution = minimize(objective, action, bounds = tuple(bnds), constraints=cons, options={'maxiter':999}, method="SLSQP")#, tol =1e-6 )
        solution = minimize(objective, action, constraints=cons, options={'maxiter':999}, method="SLSQP")#, tol =1e-6 )
        # import pdb; pdb.set_trace()
        if not solution.success:
            import pdb; pdb.set_trace()
            print("FALSE")
        # import pdb; pdb.set_trace()
        E = solution.x

        action = np.array(list(E[:3])+[1-sum(E[:3])]+ list(E[3:])+[1-sum(E[3:])])

        return (action>=0)*action
