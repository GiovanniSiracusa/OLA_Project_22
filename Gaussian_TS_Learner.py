from Learner import *
import numpy as np

class Gaussian_TS_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.sum = np.zeros(n_arms)
        self.counter = np.zeros(n_arms)

    def pull_arm(self):
        idx = np.argmax(np.random.normal((self.sum/self.counter), (1/(self.counter+1))))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.counter[pulled_arm] += 1
        self.sum[pulled_arm] += reward
        print("Pulled_arm: ",pulled_arm)
        print("Sum: ", self.sum)
        #print(self.beta_parameters)
