from simulator import *
from Step_2 import step_2
from Step_3 import step_3
from Step_4 import step_4
from Step_5 import step_5
from Step_6 import step_6
import matplotlib.pyplot as plt

def main():
    sim = Simulator(0)

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
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_3()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 4:
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_4()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 5:
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_5()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 6:
            opt_final = np.array([])
            sim1 = Simulator(1)
            sim2 = Simulator(2)
            opt, best_price_conf = sim.bruteforce()
            opt1, best_price_conf1 = sim1.bruteforce()
            opt2, best_price_conf2 = sim2.bruteforce()
            for i in range(0,150):
                opt_final=np.append(opt_final,opt)
            for i in range(150,300):
                opt_final=np.append(opt_final,opt1)
            for i in range(300,450):
                opt_final=np.append(opt_final,opt2)
            rewardsSW , rewardsCD = step_6()
            plot_regret(opt_final, rewardsSW, rewardsCD,step=6)
            break

        elif step == 7:
            break

        else:
            print("You entered an invalid step number. Please try again.")


def plot_regret(opt, rewardsTS_exp, rewardsUCB_exp,step=0):
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
    #plt.plot(np.cumsum(100*[opt]-rewards_per_experiment))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
