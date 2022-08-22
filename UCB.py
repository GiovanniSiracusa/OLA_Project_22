from Learner import Learner
import numpy as np

class UCB(Learner):
    def __init__(self,n_arms, alpha=None, items=None, graph=None):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

        if np.all(alpha != None): 
            self.alpha = alpha
        else:
            self.alpha = 1

        if np.all(items != None):
            self.items = items
        else:
            self.items = 1

        if np.all(graph != None):
            self.graph = graph
        else:
            self.graph = np.ones((n_arms, 5))

        self.arm_counter = np.zeros(n_arms)

    def pull_arm(self, margin):
        upper_conf = self.empirical_means + self.confidence
        return np.argmax(upper_conf*margin*self.alpha*self.items)
        #return np.random.choice(np.where(upper_conf == upper_conf.max()))

    def pull_arm_step5(self, margin):
        upper_conf = self.empirical_means + self.confidence
        idx = np.argmax(upper_conf*margin*np.sum(self.graph, axis=1))
        return idx

    def update(self, pulled_arm, reward, buyers, offers, alpha=None, items=None, graph=None):
        self.t +=1
        for i in range(0, buyers.astype(int)):
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + 1) / self.t
        for i in range(0, (offers.astype(int) - buyers.astype(int))):
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]* (self.t-1) + 0)/self.t

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
        self.update_observations(pulled_arm,reward)

        if alpha != None:
            self.alpha = (self.alpha * (self.t-1) + alpha)/self.t
        
        if items != None:
            self.items = (self.items * (self.t-1) + items)/self.t
        
        if np.all(graph != None):
            self.arm_counter[pulled_arm] += 1
            self.graph[pulled_arm] = (self.graph[pulled_arm]*(self.arm_counter[pulled_arm] - 1) + graph)/self.arm_counter[pulled_arm]
