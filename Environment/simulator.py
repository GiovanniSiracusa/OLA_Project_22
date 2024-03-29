import random

import numpy as np
from Environment.user_class import UserClass
import matplotlib.pyplot as plt

#np.random.seed(seed=3259414887)
#random.seed(1111)

T = 100


class Simulator:
    n_products = 5
    n_prices = 4
    l = 0.8

    def __init__(self, config_file=0):
        if config_file == 0:
            from Config import config as cf
        if config_file == 1:
            from Config import config1 as cf
        if config_file == 2:
            from Config import config3 as cf

        self.cr_mean = cf.cr_mean
        self.alphas_mean = cf.alphas_mean
        self.margin = cf.margin
        self.graph_probs_mean = cf.graph_probs_mean

        self.u1 = UserClass(
            cf.conversion_rates4,
            cf.min_daily_users4,
            cf.max_daily_users4,
            cf.alphas4,
            cf.sold_items4,
            cf.graph_probs4
        )

        self.u2 = UserClass(
            cf.conversion_rates2,
            cf.min_daily_users2,
            cf.max_daily_users2,
            cf.alphas2,
            cf.sold_items2,
            cf.graph_probs2
        )

        self.u3 = UserClass(
            cf.conversion_rates3,
            cf.min_daily_users3,
            cf.max_daily_users3,
            cf.alphas3,
            cf.sold_items3,
            cf.graph_probs3
        )

        self.user_classes = [self.u1, self.u2, self.u3]
        self.lam = cf.lambda_mat2

    def dec_to_base(self, num, base=4):  # Maximum base - 36
        base_num = ""
        while num > 0:
            dig = int(num % base)
            if dig < 10:
                base_num += str(dig)
            else:
                base_num += chr(ord('A') + dig - 10)  # Using uppercase letters
            num //= base

        base_num = base_num[::-1]  # To reverse the string

        return np.array([int(a) for a in str(base_num).zfill(self.n_products)])

    def bruteforce(self, step=0):
        max = 0
        best_conf = None
        for i in range(self.n_prices ** self.n_products):
            conf = self.dec_to_base(i)
            reward = 0
            if step==5:
                _, buyers, offers, alphas, items, history, previous = self.simulate(conf)
                graph_prob = self.estimate_probabilities(history, previous)
            #print(i)
            for p, c in enumerate(conf):
                if step == 5:
                    reward += self.cr_mean[p][c] * self.margin[p][c] * np.sum(graph_prob[p])
                else :
                    reward += self.alphas_mean[p + 1] * self.cr_mean[p][c] * self.margin[p][c]
            
            if reward > max:
                max = reward
                best_conf = conf

        opt_per_product = np.zeros(self.n_products)
        for i in range(100):
            #print("Bruteforce:", i)
            reward = self.simulate(best_conf, users=100)[0]
            opt_per_product += reward
        opt_per_product /= 100

        opt = np.sum(opt_per_product)
        print(f"Bruteforce\n Max reward: {opt}\n Best configuration: {best_conf}")
        return opt, opt_per_product, best_conf

    def estimate_probabilities(self, dataset, previous):
        credits = np.zeros((self.n_products, self.n_products))
        active = np.zeros(self.n_products)

        for episode, prev in zip(dataset, previous):
            for e, p in zip(episode, prev):
                idx = np.argwhere(e).reshape(-1)
                active[idx] += 1
                if p >= 0:
                    credits[p, idx] += 1
                else:
                    credits[idx, idx] += 0
        # print(credits)
        # print(active)
        # print(np.sum(credits, axis=0))
        for i in range(self.n_products):
            for j in range(self.n_products):
                credits[i, j] = credits[i, j] / active[i]
        # print(credits.T)
        return credits * self.lam

    def initial_node(self, alphas):
        nodes = np.array(range(self.n_products + 1))
        initial_node = np.random.choice(nodes, 1, p=alphas)
        initial_active_node = np.zeros(self.n_products + 1)
        initial_active_node[initial_node] = 1
        initial_active_node = initial_active_node[1:]

        return initial_active_node

    def simulate(self, price_conf, users=None, cl_number = -1):
        reward = np.zeros(self.n_products)
        buyers = np.zeros(self.n_products)
        offers = np.zeros(self.n_products)
        alphas = np.zeros(self.n_products)
        items = np.zeros(self.n_products)
        total_history = []
        total_previous = []

        total_users = 0
        for cl_id, cl in enumerate(self.user_classes):
            if cl_number >= 0:
                if cl_id != cl_number:
                    continue
            if users == None:
                daily_users = random.randint(cl.min_daily_users, cl.max_daily_users)
            else:
                daily_users = users
            total_users = total_users + daily_users

            for i in range(daily_users):
                # Se ci sono le alpha
                initial_active_node = self.initial_node(cl.alphas)
                # initial_active_node = self.initial_node(None)

                if all(initial_active_node == 0):
                    continue

                alphas[np.argwhere(initial_active_node).reshape(-1)] += 1

                prob_matrix = cl.graph_probs.copy() * self.lam
                np.fill_diagonal(prob_matrix, 0)

                history = np.empty((0, self.n_products))
                active_nodes_list = np.array([initial_active_node])
                previous_all = np.zeros(self.n_products, dtype=np.int8) - 2
                previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

                t = 0
                while (len(active_nodes_list) > 0):
                    active_node = active_nodes_list[0].copy()
                    active_nodes_list = active_nodes_list[1:]
                    idx_active = np.argwhere(active_node).reshape(-1)
                    # print("Active node ", active_node, end='\n')

                    # Mostra prodotto idx_active
                    offers[idx_active] += 1

                    # Quando acquista
                    if np.random.uniform(0, 1) < cl.conversion_rates[idx_active, price_conf[idx_active]]:
                        # Conta il numero di volte che un utente ha acquistato il prodotto
                        buyers[idx_active] += 1

                        # Calcola il reward per tot item comprati
                        items_sold = 1 + np.random.poisson(cl.sold_items[idx_active])
                        items[idx_active] += items_sold

                        reward[idx_active] += self.margin[idx_active,
                                                        price_conf[idx_active]] * items_sold

                        # print("idx_active ", idx_active)
                        # print(prob_matrix)
                        p = (prob_matrix.T * active_node).T
                        # print(p)
                        rnd = np.argwhere(p[idx_active] > 0)[:, 1]
                        # # print(rnd)
                        # if len(rnd) == 0:
                        #     rnd = np.array([0, 0])
                        # if len(rnd) == 1:
                        #     rnd = np.append(rnd, 0)
                        #     # np.random.choice(np.where(np.arange(self.n_products) != idx_active)[
                        #     #              0], 2, replace=False)
                        # # print("Possible choice: ", rnd)
                        # for i in range(self.n_products):
                        #     # Multiply by lambda the secondary product in the second slot
                        #     if i == rnd[0]:
                        #         pass
                        #     elif i == rnd[1]:
                        #         p[idx_active, i] = p[idx_active, i] * self.l
                        #     else:
                        #         p[idx_active, i] = 0

                        # print(p)
                        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
                        # print("Activated edges: ", activated_edges)
                        prob_matrix[:, idx_active] = 0

                        newly_active_nodes = (
                                                     np.sum(activated_edges, axis=0) > 0) * (1 - active_node)
                        # print("Newly active nodes: ", newly_active_nodes)

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

                total_history.append(history)
                total_previous.append(previous)

        # return history, previous
        items_mean = np.zeros(self.n_products)
        for i in range(self.n_products):
            if buyers[i] != 0:
                items_mean[i] = items[i] / buyers[i]
        
        return reward / total_users, buyers, offers, alphas / total_users, items_mean, total_history, total_previous


if __name__ == '__main__':
    sim = Simulator()
    opt, max_price_conf = sim.bruteforce()
    # opt_per_product, max_price_conf = sim.step2()
    # rewardsTS_exp, rewardsUCB_exp = sim.step3()
    # rewardsTS_exp, rewardsUCB_exp = sim.step4()
    # rewardsTS_exp, rewardsUCB_exp = sim.step5()
    rewardsTS_exp, rewardsUCB_exp = sim.step6()
    # print(rewardsTS_exp)

    # opt = np.sum(opt_per_product)
    print("Optimal is", opt)
    print("Max price conf", max_price_conf)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(T * [opt])
    # plt.plot(np.mean(rewardsTS_exp, axis=0),'r')
    # plt.plot(np.mean(rewardsUCB_exp, axis=0),'g')
    # plt.plot(np.cumsum(T*[opt]),'b')
    plt.plot(np.cumsum(np.mean(opt - rewardsTS_exp, axis=0)), 'r')
    plt.plot(np.cumsum(np.mean(opt - rewardsUCB_exp, axis=0)), 'g')
    # plt.plot(np.cumsum(100*[opt]-rewards_per_experiment))
    plt.show()
