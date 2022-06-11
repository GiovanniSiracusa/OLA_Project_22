import numpy as np

np.random.seed(seed=456)


# DEFINIRE CONVERSION RATES PER OGNI CLASSE DI UTENTI
#CATEGORIA PRODOROTTI: ARTICOLI PER LA CASA
#MASCHI <30 ANNI
conversion_rates1 = np.array([[0.9, 0.6, 0.5, 0.3, 0.1],    #Alexa
                             [0.5, 0.4, 0.3, 0.1, 0.01],    #Quadro medium/high price
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Cantinetta per il vino
                             [0.4, 0,3, 0.2, 0.05, 0.01],   #Pianta grassa
                             [0.6, 0.45, 0.3, 0.1, 0.05]])  #Pouf
#FEMMINE <30 ANNI
conversion_rates2 = np.array([[0.6, 0.45, 0.3, 0.1, 0.05],  #Alexa
                             [0.5, 0.4, 0.3, 0.1, 0.01],    #Quadro medium/high price
                             [0.4, 0,3, 0.2, 0.05, 0.01],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Pianta grassa
                             [0.4, 0.35, 0.3, 0.25, 0.2]])  #Pouf
#MASCHI >30 ANNI
conversion_rates3 = np.array([[0.9, 0.7, 0.5, 0.4, 0.2],    #Alexa
                             [0.5, 0.3, 0.2, 0.1, 0.01],    #Quadro medium/high price
                             [0.9, 0.7, 0.4, 0.2, 0.1],     #Cantinetta per il vino
                             [0.3, 0.25, 0.2, 0.1, 0.05],   #Pianta grassa
                             [0.4, 0.35, 0.3, 0.25, 0.2]])  #Pouf
#FEMMINE >30
conversion_rates4 = np.array([[0.9, 0.7, 0.5, 0.4, 0.2],    #Alexa
                             [0.5, 0.3, 0.2, 0.1, 0.01],    #Quadro medium/high price
                             [0.9, 0.7, 0.4, 0.2, 0.1],     #Cantinetta per il vino
                             [0.3, 0.25, 0.2, 0.1, 0.05],   #Pianta grassa
                             [0.4, 0.35, 0.3, 0.25, 0.2]])  #Pouf

# DEFINIRE I PREZZI DI OGNI PRODOTTO
prices = np.array([[100, 500, 1500, 3000, 10000],
                  [10, 12, 15, 18, 20],
                  [10, 12, 15, 18, 20],
                  [10, 12, 15, 18, 20],
                  [10, 12, 15, 18, 20]])

# DEFINIRE ALPHAS PER OGNI CLASSE DI UTENTI
alphas1 = np.array([0.1, 0.18, 0.18, 0.18, 0.18, 0.18])  #ELEMENTO 0 -> Non entra nel sito
alphas2 = np.array([0.1, 0.18, 0.18, 0.18, 0.18, 0.18])  #LA SOMMA DELLE ALPHA E' 1
alphas3 = np.array([0.1, 0.18, 0.18, 0.18, 0.18, 0.18])

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
graph_probs1 = np.array([[0, 0.5, 0.3, 0.15, 0.1],
                        [0.5, 0, 0.2, 0.1, 0.01],
                        [0.9, 0.7, 0, 0.2, 0.1],
                        [0.3, 0.25, 0.2, 0, 0.05],
                        [0.4, 0.35, 0.3, 0.25, 0]])
graph_probs2 = np.random.rand(5, 5)
graph_probs3 = np.random.rand(5, 5)
