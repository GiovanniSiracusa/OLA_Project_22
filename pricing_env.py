import random
import numpy as np
import config as cf

class PricingEnv:
    def __init__(self):
        self.matrix = []

        self.matrix.append(UserClass(
            cf.conversion_rates1, 
            cf.min_daily_users1, 
            cf.max_daily_users1, 
            cf.alphas1, 
            cf.sold_items1, 
            cf.graph_probs1
            )
        )

        self.matrix.append(UserClass(
            cf.conversion_rates2, 
            cf.min_daily_users2, 
            cf.max_daily_users2, 
            cf.alphas2, 
            cf.sold_items2, 
            cf.graph_probs2
            )
        )

        self.matrix.append(UserClass(
            cf.conversion_rates3, 
            cf.min_daily_users3, 
            cf.max_daily_users3, 
            cf.alphas3, 
            cf.sold_items3, 
            cf.graph_probs3
            )
        )

class UserClass:
    def __init__(self, conversion_rates, min_daily_users, max_daily_users, alphas, sold_items, graph_probs): 
        self.conversion_rates = conversion_rates # conversion rate []
        self.min_daily_users = min_daily_users
        self.max_daily_users = max_daily_users
        self.alphas = alphas
        self.sold_items = sold_items
        self.graph_probs = graph_probs

    def get_daily_users(self):
        return random.randint(self.min_daily_users, self.max_daily_users)
    
    def get_sold_items(self):
        return random.randint(1, self.max_sold_items)