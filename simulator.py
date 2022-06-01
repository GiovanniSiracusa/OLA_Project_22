import config as cf
import numpy as np
from pricing_env import UserClass

class Simulator:
    def step1(self):
        u1 = UserClass(
            cf.conversion_rates1,
            cf.min_daily_users1,
            cf.max_daily_users1,
            cf.alphas1,
            cf.max_sold_items1,
            cf.graph_probs1
        )

        u2 = UserClass(
            cf.conversion_rates2,
            cf.min_daily_users2,
            cf.max_daily_users2,
            cf.alphas2,
            cf.max_sold_items2,
            cf.graph_probs2
        )

        u3 = UserClass(
            cf.conversion_rates3,
            cf.min_daily_users3,
            cf.max_daily_users3,
            cf.alphas3,
            cf.max_sold_items3,
            cf.graph_probs3
        )

        self.simulate(user_classes=[u1, u2, u3])

    def simulate(self, user_classes):
        ''' Per ogni classe prendo N utenti e'''
