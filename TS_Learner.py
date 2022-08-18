from Learner import *
import numpy as np

class TS_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms,2))

    def pull_arm(self,prices):
        b = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        #print(b*prices)
        idx = np.argmax(b*prices)
        return idx

    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
        #print("pulled_arm", pulled_arm)
        #print(self.beta_parameters)

    def plot_distribution(self):
        from scipy.stats import beta
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        colors=['r', 'g', 'b', 'm']
        for i in range(self.n_arms):
            a1, b1 = self.beta_parameters[i, 0], self.beta_parameters[i, 1]
            #mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
            x = np.linspace(beta.ppf(0.01, a1, b1),
                    beta.ppf(0.99, a1, b1), 100)
            ax.plot(x, beta.pdf(x, a1, b1),
                colors[i], lw=0.5, alpha=0.6, label='beta pdf')

