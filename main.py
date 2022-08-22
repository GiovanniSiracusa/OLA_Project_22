from simulator import *
import matplotlib.pyplot as plt

def main():
    sim = Simulator()

    while True:
        step = int(input("Select the step [2-3-4-5-6-7]: "))

        if step == 2:
            opt, best_price_conf = sim.bruteforce()
            opt_per_product, max_price_conf = sim.step2()
            print("\nRevenue provided by the greedy algorithm:", np.sum(opt_per_product),
                  "\nOptimal price configuration", max_price_conf)
            print("\nRevenue provided by the brute force algorithm:", opt,
                  "\nOptimal price configuration", best_price_conf)
            break

        elif step == 3:
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = sim.step3()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 4:
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = sim.step4()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 5:
            opt, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = sim.step5()
            plot_regret(opt, rewardsTS, rewardsUCB)
            break

        elif step == 6:
            break

        elif step == 7:
            break

        else:
            print("You entered an invalid step number. Please try again.")


def plot_regret(opt, rewardsTS, rewardsUCB):
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")
    # plt.plot(T*[opt])
    # plt.plot(rewardsTS,'r')
    # plt.plot(rewardsUCB,'g')
    # plt.plot(np.cumsum(T*[opt]),'b')
    plt.plot(np.cumsum(opt - rewardsTS), 'r')
    plt.plot(np.cumsum(opt - rewardsUCB), 'g')
    # plt.plot(np.cumsum(100*[opt]-rewards_per_experiment))
    plt.show()


if __name__ == "__main__":
    main()
