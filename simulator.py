import random
from typing import List
from Greedy_Learner import Greedy_Learner
import config as cf
import numpy as np
from pricing_env import UserClass

class Simulator:
    n_nodes = 5
    l = 0.8

    def step2(self):
        u1 = UserClass(
            cf.conversion_rates1,
            cf.min_daily_users1,
            cf.max_daily_users1,
            cf.alphas1,
            cf.max_sold_items1,
            cf.graph_probs1
        )

        u2 = UserClass(
            cf.conversion_rates2,
            cf.min_daily_users2,
            cf.max_daily_users2,
            cf.alphas2,
            cf.max_sold_items2,
            cf.graph_probs2
        )

        u3 = UserClass(
            cf.conversion_rates3,
            cf.min_daily_users3,
            cf.max_daily_users3,
            cf.alphas3,
            cf.max_sold_items3,
            cf.graph_probs3
        )

        learner = Greedy_Learner(self.n_nodes)

        time_horizon = 1000

        
        temp_id = -1
        max_reward = 0
        temp_max = 0
        max_price_conf = np.zeros(self.n_nodes, dtype=np.int8)
        temp_max_conf = max_price_conf
        counter = -1

        price_conf_history = np.empty((0, self.n_nodes))

        for t in range(time_horizon):
            price_conf = learner.pull_arm(counter, max_price_conf)
            
            if not price_conf.tolist() in price_conf_history.tolist():
                price_conf_history = np.concatenate((price_conf_history, [price_conf]), axis=0)
                reward = 0
                for cl in [u1,u2,u3]:
                    daily_users = random.randint(cl.min_daily_users, cl.max_daily_users)
                    for i in range(daily_users):
                        n = self.simulate(cl, price_conf)
                        reward = reward + n
                print(price_conf, reward)
                # trova un nuovo max
                if counter == -1:
                    max_reward = reward
                    max_price_conf = price_conf
                    learner.update(temp_id, max_reward)
                elif reward > temp_max:
                    temp_max = reward
                    temp_max_conf = price_conf
                    temp_id = counter

            counter += 1

            if counter == self.n_nodes:
                counter = 0
                if temp_max >= max_reward:
                    max_reward = temp_max
                    max_price_conf = temp_max_conf
                    temp_max = 0
                    learner.update(temp_id, max_reward)
                    print(temp_id, max_reward)
                else:
                    break
        print("Max price conf:", max_price_conf, "Max reward:", max_reward)
                    
    
    def initial_node(self, alphas):
        nodes = np.array(range(self.n_nodes + 1))
        initial_node = np.random.choice(nodes, 1, p=alphas)
        initial_active_node = np.zeros(self.n_nodes + 1)
        initial_active_node[initial_node] = 1
        initial_active_node = initial_active_node[1:]

        return initial_active_node

    def simulate(self, uc: UserClass, price_conf):
        initial_active_node = self.initial_node(uc.alphas)

        if all(initial_active_node == 0):
            return 0

        prob_matrix = uc.graph_probs.copy()
        np.fill_diagonal(prob_matrix, 0)
        
        history = np.empty((0, self.n_nodes))
        active_nodes_list = np.array([initial_active_node])
        previous_all = np.zeros(self.n_nodes, dtype=np.int8)-2
        previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

        reward = 0

        t = 0
        while (len(active_nodes_list) > 0):
            active_node = active_nodes_list[0].copy()
            active_nodes_list = active_nodes_list[1:]
            idx_active = np.argwhere(active_node).reshape(-1)
            #print("Active node ", active_node, end='\n')
            
            if np.random.uniform(0, 1) < uc.conversion_rates[idx_active, price_conf[idx_active]]: # Quando acquista

                # Calcola il reward per tot item comprati
                items_sold = random.randint(1, uc.max_sold_items)
                reward = reward + cf.margin[idx_active, price_conf[idx_active]] * items_sold

                p = (prob_matrix.T * active_node).T
                rnd = np.random.choice(np.where(np.arange(self.n_nodes) != idx_active)[0], 2, replace=False)
                #print("Possible choice: ", rnd)
                for i in range(self.n_nodes):
                    # Multiply by lambda the secondary product in the second slot
                    if i == rnd[0]:
                        pass
                    elif i ==  rnd[1]:
                        p[idx_active,i] = p[idx_active, i] * self.l
                    else:
                        p[idx_active, i] = 0

                #print(p)
                activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
                #print("Activated edges: ", activated_edges)
                prob_matrix[:, idx_active] = 0
                
                newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_node)
                #print("Newly active nodes: ", newly_active_nodes)
                # Split newly active nodes

                for idx in rnd:
                    if newly_active_nodes[idx] == 1:
                        prob_matrix[:, idx] = 0
                        a = np.zeros(5)
                        a[idx] = 1
                        active_nodes_list = np.concatenate((active_nodes_list, [a]), axis=0)
                        previous_all[idx] = idx_active
                #print(active_nodes_list)
            history = np.concatenate((history, [active_node]), axis=0)

        previous = np.array([], dtype=np.int8)
        for e in history:
            previous = np.append(previous, previous_all[np.argwhere(e).reshape(-1)])
        #return history, previous
        return reward



sim = Simulator()
sim.step2()