import time
from simulator import *
from Step_2 import step_2
from Step_3 import step_3
from Step_4 import step_4
from Step_5 import step_5
from Step_6 import step_6
from Step_7 import step_7
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def main():
    sim = Simulator(0)
    time_horizon = 300
    while True:
        step = int(input("Select the step [2-3-4-5-6-7]: "))

        if step == 2:
            opt, best_price_conf = sim.bruteforce()
            opt_per_product, max_price_conf = step_2()
            print("\nRevenue provided by the greedy algorithm:", np.sum(opt_per_product),
                  "\nOptimal price configuration", max_price_conf)
            print("\nRevenue provided by the brute force algorithm:", opt,
                  "\nOptimal price configuration", best_price_conf)
            break

        elif step == 3:
            opt,opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB, mean_rewards = step_3(time_horizon)
            bound = compute_UCBbound(opt_per_product,mean_rewards, time_horizon)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon, bound)
            break

        elif step == 4:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_4(time_horizon)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon)
            break

        elif step == 5:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_5(time_horizon)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon)
            break

        elif step == 6:
            opt_final = np.array([])
            sim1 = Simulator(1)
            sim2 = Simulator(2)
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            opt1, opt_per_product1, best_price_conf = sim1.bruteforce()
            opt2, opt_per_product2, best_price_conf = sim2.bruteforce()
            for i in range(0,150):
                opt_final=np.append(opt_final,opt)
            for i in range(150,300):
                opt_final=np.append(opt_final,opt1)
            for i in range(300,450):
                opt_final=np.append(opt_final,opt2)
            rewardsSW , rewardsCD = step_6()
            plot_regret(opt_final, rewardsSW, rewardsCD, 450, step=6)
            break

        elif step == 7:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_7(time_horizon)
            # plot_regret(opt, rewardsTS, rewardsUCB, time_horizon)
            plot_reward(opt, rewardsTS, rewardsUCB, time_horizon)
            break

        else:
            print("You entered an invalid step number. Please try again.")


def plot_regret(opt, rewardsTS_exp, rewardsUCB_exp, time_horizon, bound = 0, step=0):
    if step==6:
        labels=["SW","CD"]
    else:
        labels=["TS","UCB"]
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")
    #plt.plot(T*[opt])
    #plt.plot(np.mean(rewardsTS_exp, axis=0),'r')
    #plt.plot(np.mean(rewardsUCB_exp, axis=0),'g')
    #plt.plot(np.cumsum(T*[opt]),'b')
    plt.plot(np.cumsum(np.mean(opt-rewardsTS_exp, axis=0)),'r',label=labels[0])
    plt.plot(np.cumsum(np.mean(opt-rewardsUCB_exp, axis=0)),'g',label=labels[1])
    if bound != 0:
        plt.plot(time_horizon*[bound], 'b')
    #plt.plot(np.cumsum(100*[opt]-rewards_per_experiment))

    #x = np.arange(len(rewardsTS_exp[0]), dtype=float)
    #y_ts=(np.cumsum(np.mean(opt-rewardsTS_exp, axis=0)))
    #y_ucb=(np.cumsum(np.mean(opt-rewardsUCB_exp, axis=0)))
    #dev_ts=np.std(opt - rewardsTS_exp, axis=0)
    #dev_ucb=np.std(opt - rewardsUCB_exp, axis=0)

    plt.legend()
    plt.show()

def plot_reward(opt, rewardsTS_exp, rewardsUCB_exp, time_horizon):
    plt.plot(time_horizon*[opt],'b')

    window = 10

    average_y = moving_average(np.mean(rewardsTS_exp, axis=0), window)
    plt.plot(average_y[:-10], 'y', label='TS_avg')
    #plt.plot(np.mean(rewardsTS_exp, axis=0),'r', label='TS')

    average_y = moving_average(np.mean(rewardsUCB_exp, axis=0), window)
    plt.plot(average_y[:-10], 'c', label='UCB_avg')
    #plt.plot(np.mean(rewardsUCB_exp, axis=0),'g', label='UCB')

    plt.legend()
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def compute_UCBbound(opt_per_product, mean_rewards, time_horizon):
    s = 0
    for i in range(5):
        for j in range(4):
            delta = opt_per_product[i] - mean_rewards[i][j]
            if delta > 0:
                s += (4*np.log(time_horizon)/delta + 8*delta)
    return s




if __name__ == "__main__":
    main()
