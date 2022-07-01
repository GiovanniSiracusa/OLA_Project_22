import random
from typing import List

from Gaussian_TS_Learner import Gaussian_TS_Learner
from Greedy_Learner import Greedy_Learner
from TS_Learner import TS_Learner
import config as cf
import numpy as np
from pricing_env import UserClass
import matplotlib.pyplot as plt




class Simulator:
    n_products = 5
    n_prices = 4
    l = 0.8
    u1 = UserClass(
        cf.conversion_rates4,
        cf.min_daily_users4,
        cf.max_daily_users4,
        cf.alphas4,
        cf.max_sold_items4,
        cf.graph_probs4
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

    user_classes = [u1, u2, u3]

    def step2(self):
        n_experiments = 1
        time_horizon = 100
        final_max_reward = 0
        final_max_price_conf = np.zeros(self.n_products, dtype=np.int8)

        for e in range(n_experiments):
            learner = Greedy_Learner(self.n_prices, self.n_products)

            print("Exp:", e)
            temp_id = -1
            max_reward = 0
            temp_max = 0
            max_price_conf = np.zeros(self.n_products, dtype=np.int8)
            temp_max_conf = max_price_conf
            counter = -1
            price_conf_history = np.empty((0, self.n_products))

            for t in range(time_horizon):
                price_conf = learner.pull_arm(counter, max_price_conf)

                if not price_conf.tolist() in price_conf_history.tolist():
                    price_conf_history = np.concatenate(
                        (price_conf_history, [price_conf]), axis=0)
                    reward, cr = self.simulate(price_conf)
                    # print(price_conf, np.round(reward,2), np.sum(reward))
                    # trova un nuovo max
                    if counter == -1:
                        max_reward = reward
                        max_price_conf = price_conf
                        learner.update()
                    elif np.sum(reward) > np.sum(temp_max):
                        temp_max = reward
                        temp_max_conf = price_conf
                        temp_id = counter

                counter += 1

                if counter == self.n_products:
                    counter = 0
                    if np.sum(temp_max) >= np.sum(max_reward):
                        max_reward = temp_max
                        max_price_conf = temp_max_conf
                        temp_max = 0
                        learner.update()
                        # print(temp_id, np.round(max_reward,2), np.sum(max_reward))
                    else:
                        break
            print("Max price conf:", max_price_conf, "Max reward:", max_reward, "Max total:", np.sum(max_reward))
            print()

            if np.sum(max_reward) > np.sum(final_max_reward):
                final_max_price_conf = max_price_conf
                final_max_reward = max_reward
        print(
            f"Final max reward {final_max_reward}\nFinal total reward {np.sum(final_max_reward)}\nFinal price conf {final_max_price_conf}\n")
        return final_max_reward


    # Pullare un arm per ogni prodotto -> [0, 2, 1, 4, 0]
    # Simulare un round con la price conf pullata -> simulate(price_conf=[0, 2, 1, 4, 0])
    # Per ogni prodotto ritornare se è stato comprato o no -> [1, 0, 0, 1, 0]
    # E calcolare il reward di ogni prodotto
    # Aggiornare le beta
    
    def step3(self, opt):
        n_experiments = 1
        time_horizon = 100

        for e in range(n_experiments):
            #learners = [TS_Learner(self.n_prices) for i in range(self.n_products)]
            learners = [Gaussian_TS_Learner(self.n_prices) for i in range(self.n_products)]
            print("Exp:", e)

            rewards = np.array([])

            for t in range(time_horizon):
                price_conf = np.array([learners[i].pull_arm() for i in range(self.n_products)])
                reward, cr = self.simulate(price_conf)
                for p in range(self.n_products):
                    ''' Non dobbiamo passare cr al TSLearner ma il reward normalizzato in base al max reward del prodotto'''
                    learners[p].update(price_conf[p], np.clip(reward[p]/opt[p], 0, 1))
                rewards = np.append(rewards, np.sum(reward))
                print(rewards)
                #print(price_conf, reward, cr)
            print("Rewards", rewards)
        return rewards

    def initial_node(self, alphas):
        nodes = np.array(range(self.n_products + 1))
        initial_node = np.random.choice(nodes, 1, p=alphas)
        initial_active_node = np.zeros(self.n_products + 1)
        initial_active_node[initial_node] = 1
        initial_active_node = initial_active_node[1:]

        return initial_active_node

    def simulate(self, price_conf, alphas=None):
        reward = np.zeros(self.n_products)
        buyers = np.zeros(self.n_products)
        offers = np.zeros(self.n_products)
        
        for cl in self.user_classes:
            daily_users = random.randint(cl.min_daily_users, cl.max_daily_users)
            for i in range(daily_users):
                # Se ci sono le alpha
                if alphas:
                    initial_active_node = self.initial_node(alphas)
                else:
                    initial_active_node = self.initial_node(cl.alphas)

                if all(initial_active_node == 0):
                    continue

                prob_matrix = cl.graph_probs.copy()
                np.fill_diagonal(prob_matrix, 0)

                history = np.empty((0, self.n_products))
                active_nodes_list = np.array([initial_active_node])
                previous_all = np.zeros(self.n_products, dtype=np.int8)-2
                previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

                t = 0
                while (len(active_nodes_list) > 0):
                    active_node = active_nodes_list[0].copy()
                    active_nodes_list = active_nodes_list[1:]
                    idx_active = np.argwhere(active_node).reshape(-1)
                    #print("Active node ", active_node, end='\n')

                    # Mostra prodotto idx_active
                    offers[idx_active] += 1

                    # Quando acquista
                    if np.random.uniform(0, 1) < cl.conversion_rates[idx_active, price_conf[idx_active]]:
                        # Conta il numero di volte che un utente ha acquistato il prodotto
                        buyers[idx_active] += 1

                        # Calcola il reward per tot item comprati
                        items_sold = random.randint(1, cl.max_sold_items)
                        reward[idx_active] += cf.margin[idx_active,
                                                    price_conf[idx_active]] * items_sold

                        p = (prob_matrix.T * active_node).T
                        rnd = np.random.choice(np.where(np.arange(self.n_products) != idx_active)[
                                            0], 2, replace=False)
                        #print("Possible choice: ", rnd)
                        for i in range(self.n_products):
                            # Multiply by lambda the secondary product in the second slot
                            if i == rnd[0]:
                                pass
                            elif i == rnd[1]:
                                p[idx_active, i] = p[idx_active, i] * self.l
                            else:
                                p[idx_active, i] = 0

                        # print(p)
                        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
                        #print("Activated edges: ", activated_edges)
                        prob_matrix[:, idx_active] = 0

                        newly_active_nodes = (
                            np.sum(activated_edges, axis=0) > 0) * (1 - active_node)
                        #print("Newly active nodes: ", newly_active_nodes)

                        # Split newly active nodes
                        for idx in rnd:
                            if newly_active_nodes[idx] == 1:
                                prob_matrix[:, idx] = 0
                                a = np.zeros(5)
                                a[idx] = 1
                                active_nodes_list = np.concatenate(
                                    (active_nodes_list, [a]), axis=0)
                                previous_all[idx] = idx_active
                        # print(active_nodes_list)
                    history = np.concatenate((history, [active_node]), axis=0)

                previous = np.array([], dtype=np.int8)
                for e in history:
                    previous = np.append(
                        previous, previous_all[np.argwhere(e).reshape(-1)])
        # return history, previous
        return reward, buyers/offers


sim = Simulator()
opt_per_product = sim.step2()
print("Optimal is", opt_per_product)
rewards_per_experiment = sim.step3(opt_per_product)

opt = np.sum(opt_per_product)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(100*[opt*100])
plt.plot(np.cumsum(rewards_per_experiment),'r')
#plt.plot(np.cumsum(opt-rewards_per_experiment))
plt.show()