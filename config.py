import numpy as np

np.random.seed(seed=1234)


# DEFINIRE CONVERSION RATES PER OGNI CLASSE DI UTENTI
conversion_rates1 = np.array([0.7, 0.5, 0.3, 0.15, 0.1],
                             [0.5, 0.3, 0.2, 0.1, 0.01],
                             [0.9, 0.7, 0.4, 0.2, 0.1],
                             [0.3, 0.25, 0.2, 0.1, 0.05],
                             [0.4, 0.35, 0.3, 0.25, 0.2])
conversion_rates2 = np.random.rand(5, 5)
conversion_rates3 = np.random.rand(5, 5)

# DEFINIRE I PREZZI DI OGNI PRODOTTO
prices1 = np.array([10, 12, 15, 18, 20])
prices2 = np.array([10, 12, 15, 18, 20])
prices3 = np.array([10, 12, 15, 18, 20])
prices4 = np.array([10, 12, 15, 18, 20])
prices5 = np.array([10, 12, 15, 18, 20])

# DEFINIRE ALPHAS PER OGNI CLASSE DI UTENTI
alphas1 = np.array([0.1, 0.16, 0.16, 0.16, 0.16, 0.16])  #ELEMENTO 0 -> Non entra nel sito
alphas2 = np.array([0.1, 0.16, 0.16, 0.16, 0.16, 0.16])  #LA SOMMA DELLE ALPHA E' 1
alphas3 = np.array([0.1, 0.16, 0.16, 0.16, 0.16, 0.16])

min_daily_users1 = 100
max_daily_users1 = 500
min_daily_users2 = 100
max_daily_users2 = 500
min_daily_users3 = 100
max_daily_users3 = 500

max_sold_items1 = 5
max_sold_items2 = 5
max_sold_items3 = 5

# DEFINIRE PROBABILITA DI PASSARE DA UN PRODOTTO A UN ALTRO PER OGNI CLASSE DI UTENTI
graph_probs1 = np.array([0, 0.5, 0.3, 0.15, 0.1],
                        [0.5, 0, 0.2, 0.1, 0.01],
                        [0.9, 0.7, 0, 0.2, 0.1],
                        [0.3, 0.25, 0.2, 0, 0.05],
                        [0.4, 0.35, 0.3, 0.25, 0])
graph_probs2 = np.random.rand(5, 5)
graph_probs3 = np.random.rand(5, 5)
