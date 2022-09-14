import numpy as np
import config as cf
from simulator import Simulator
from Greedy_Learner import Greedy_Learner


def step_2():
    sim=Simulator(config_file=0)
    n_experiments = 1
    time_horizon = 100
    final_max_reward = 0
    final_max_price_conf = np.zeros(sim.n_products, dtype=np.int8)

    for e in range(n_experiments):
        learner = Greedy_Learner(sim.n_prices, sim.n_products)

        #print("Exp:", e)
        temp_id = -1
        max_reward = 0
        temp_max = 0
        max_price_conf = np.zeros(sim.n_products, dtype=np.int8)
        temp_max_conf = max_price_conf
        counter = -1
        price_conf_history = np.empty((0, sim.n_products))

        for t in range(time_horizon):
            price_conf = learner.pull_arm(counter, max_price_conf)
            reward = 0
            if not price_conf.tolist() in price_conf_history.tolist():
                price_conf_history = np.concatenate(
                    (price_conf_history, [price_conf]), axis=0)
                for i, j in enumerate(price_conf):
                    reward += cf.margin[i, j] * cf.cr_mean[i, j] * cf.alphas_mean[i + 1]
                    # print(reward)
                #print(price_conf, np.round(reward, 2), np.sum(reward))
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

            if counter == sim.n_products:
                counter = 0
                if np.sum(temp_max) >= np.sum(max_reward):
                    max_reward = temp_max
                    max_price_conf = temp_max_conf
                    temp_max = 0
                    learner.update()
                    # print(temp_id, np.round(max_reward,2), np.sum(max_reward))
                else:
                    break
        # print("Max price conf:", max_price_conf, "Max reward:", max_reward, "Max total:", np.sum(max_reward))
        #print()

        if np.sum(max_reward) > np.sum(final_max_reward):
            final_max_price_conf = max_price_conf
            final_max_reward = max_reward

    opt_reward = 0
    for i in range(100):
        #print("Greedy:", i)
        reward = sim.simulate(final_max_price_conf, users=100)[0]
        opt_reward += reward
    opt_reward /= 100

    #print(
    #    f"Final max reward {opt_reward}\nOpt Reward {np.sum(opt_reward)}\n\nFinal price conf {final_max_price_conf}\n")

    return opt_reward, final_max_price_conf