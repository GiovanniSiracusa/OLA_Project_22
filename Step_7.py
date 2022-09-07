import numpy as np
import config as cf
from TS_Learner import TS_Learner
from UCB import UCB
from simulator import Simulator


def step_7(time_horizon):
    n_experiments = 1
    #time_horizon = 100
    sim = Simulator(0)

    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        # learners = [TS_Learner(sim.n_prices) for i in range(sim.n_products)]
        ts = [[TS_Learner(sim.n_prices) for i in range(sim.n_products)] for j in range(3)]
        ucb = [[UCB(sim.n_prices) for i in range(sim.n_products)] for j in range(3)]

        print("Exp:", e)

        rewardsTS = np.array([])
        rewardsUCB = np.array([])

        for cl in range(3):
            for t in range(time_horizon):
                # TS Learner
                price_conf = np.array([ts[cl][i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
                reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf, cl_number=cl)
                for p in range(sim.n_products):
                    ts[cl][p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
                rewardsTS = np.append(rewardsTS, np.sum(reward))
                print(t)
                print("TS: ", price_conf)
            # print("Reward: ", reward)
            # print(price_conf, reward, cr
        rewardsTS = np.mean(np.reshape(rewardsTS, (3, -1)), axis=0)
        
        for cl in range(3):
            for t in range(time_horizon):
                # UCB
                price_conf = np.array([ucb[cl][i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
                reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf, cl_number=cl)
                for p in range(sim.n_products):
                    ucb[cl][p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
                rewardsUCB = np.append(rewardsUCB, np.sum(reward))
                # print(t)
                print("UCB: " ,price_conf)

                # print("Reward: ", reward)
        rewardsUCB = np.mean(np.reshape(rewardsUCB, (3, -1)), axis=0)

        for i in range(5):
            print(i)
            print("Alpha:", ts[0][i].alpha)
            print("Items:", ts[0][i].items)

        rewardsTS_exp.append(rewardsTS)
        rewardsUCB_exp.append(rewardsUCB)

        # print("Rewards", rewardsTS)
        # print("Rewards", rewardsUCB)
    return rewardsTS_exp, rewardsUCB_exp