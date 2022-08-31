import numpy as np
import config as cf
from TS_Learner import TS_Learner
from UCB import UCB
from simulator import Simulator


def step_5():
    n_experiments = 1
    time_horizon = 300
    sim = Simulator(0)
    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        # learners = [TS_Learner(sim.n_prices) for i in range(sim.n_products)]
        ts = [TS_Learner(sim.n_prices, cf.alphas_mean[i], cf.sold_items_mean[i]) for i in range(sim.n_products)]
        ucb = [UCB(sim.n_prices) for i in range(sim.n_products)]

        print("Exp:", e)

        rewardsTS = np.array([])
        rewardsUCB = np.array([])

        for t in range(time_horizon):
            # TS Learner
            price_conf = np.array([ts[i].pull_arm_step5(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, alphas, items, history, previous = sim.simulate(price_conf)
            graph_prob = sim.estimate_probabilities(history, previous)
            # print(graph_prob)

            for p in range(sim.n_products):
                ts[p].update(price_conf[p], reward[p], buyers[p], offers[p], graph=graph_prob[p])
            rewardsTS = np.append(rewardsTS, np.sum(reward))
            print(t)
            print("TS: ", price_conf)
            # print("Reward: ", reward)
            # print(price_conf, reward, cr)

        for t in range(time_horizon):
            # UCB
            price_conf = np.array([ucb[i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, alphas, items, history, previous = sim.simulate(price_conf)
            for p in range(sim.n_products):
                ucb[p].update(price_conf[p], reward[p], buyers[p], offers[p], graph=graph_prob[p])
            rewardsUCB = np.append(rewardsUCB, np.sum(reward))
            # print(t)
            print("UCB: ", price_conf)

            # print("Reward: ", reward)

        rewardsTS_exp.append(rewardsTS)
        rewardsUCB_exp.append(rewardsUCB)
        # print("Rewards", rewardsTS)
        # print("Rewards", rewardsUCB)
    return rewardsTS_exp, rewardsUCB_exp