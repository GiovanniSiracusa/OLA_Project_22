import numpy as np
from scipy.optimize import linear_sum_assignment
from CUSUM import CUSUM
from UCB import UCB

class CD_UCB(UCB):
    def __init__(self, n_arms, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self, margin):
        if np.random.binomial(1, 1-self.alpha):
            return super().pull_arm(margin)
        else:
            return np.random.randint(0, 4)

    def update(self, pulled_arm, reward, buyers, offers):
        self.t +=1
        if self.change_detection[pulled_arm].update(reward):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arms[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
            print("!!!!!!!!!!CHANGE DETECTED!!!!!!!", self.t)
        self.update_observations(pulled_arm, reward, buyers, offers)
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward, buyers, offers):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(buyers/offers)
        self.collected_rewards = np.append(self.collected_rewards, reward)
