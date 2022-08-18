from Learner import Learner
import numpy as np

class UCB(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def pull_arm(self,prices):
        upper_conf = self.empirical_means + self.confidence
        return np.argmax(upper_conf*prices)
        #return np.random.choice(np.where(upper_conf == upper_conf.max()))

    def update(self, pull_arm,reward, buyers,offers):
        self.t +=1
        for i in range(0, buyers.astype(int)):
            self.empirical_means[pull_arm] = (self.empirical_means[pull_arm] * (self.t - 1) + 1) / self.t
        for i in range(0, (offers.astype(int) - buyers.astype(int))):
            self.empirical_means[pull_arm] = (self.empirical_means[pull_arm]* (self.t-1) + 0)/self.t

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
        self.update_observations(pull_arm,reward)
