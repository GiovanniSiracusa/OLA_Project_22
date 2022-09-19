import numpy as np


# DEFINIRE CONVERSION RATES PER OGNI CLASSE DI UTENTI

#CATEGORIA PRODOROTTI: ARTICOLI PER LA CASA
#Prodotti associati a categorie:
#PERSONE <30  --> POUF
#MASCHI >30 --> CANTINETTA PER IL VINI
#FEMMINE >30 --> PIANTA GRASSA
#QUADRO MEDIUM/HIGH PRICE PIACE A TUTTE LE CATEGORIE MA MENO DEI PREFERITI

#uomini e donna < 30 unica categoria

#PERSONE <30 ANNI
conversion_rates2 = np.array([[0.6, 0.45, 0.3, 0.1],  #Alexa
                             [0.5, 0.4, 0.3, 0.1],    #Quadro medium/high price
                             [0.4, 0.3, 0.2, 0.05],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1],   #Pianta grassa
                             [0.9, 0.75, 0.6, 0.3]])  #Pouf
#MASCHI >30 ANNI
conversion_rates3 = np.array([[0.6, 0.45, 0.3, 0.1],  #Alexa
                             [0.5, 0.3, 0.2, 0.1],    #Quadro medium/high price
                             [0.9, 0.75, 0.6, 0.3],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1],   #Pianta grassa
                             [0.4, 0.3, 0.2, 0.05]])  #Pouf
#FEMMINE >30 ANNI
conversion_rates4 = np.array([[0.4, 0.3, 0.2, 0.05],  #Alexa
                             [0.5, 0.3, 0.2, 0.1],    #Quadro medium/high price
                             [0.6, 0.45, 0.3, 0.1],   #Cantinetta per il vino
                             [0.9, 0.75, 0.6, 0.3],    #Pianta grassa
                             [0.6, 0.45, 0.3, 0.1]])  #Pouf

cr_mean = (conversion_rates2+conversion_rates3+conversion_rates4)/3

# DEFINIRE I PREZZI DI OGNI PRODOTTO
prices = np.array([[20, 25, 34, 50],         #alexa
                  [169, 189, 205, 215],      #quadro
                  [230, 240, 250, 330],      #cantinetta
                  [16, 20, 23, 26],          #pianta
                  [65, 70, 76, 80]])         #pouf

#alexa, quadro, cantinetta, pianta, pouf
costs = np.array([[16], [88], [152], [4], [25]])

margin = prices - costs

# DEFINIRE ALPHAS PER OGNI CLASSE DI UTENTI
# assunzione i maschi hanno meno probabilità di entrare in un sito di arredamento per la casa
# np.array([no sito, alexa, quadro, cantinetta, pianta, pouf])
# ELEMENTO 0 -> Non entra nel sito
alphas2 = np.array([0.1, 0.185, 0.11, 0.06, 0.185, 0.36])    #PERSONE <30             LA SOMMA DELLE ALPHA E' 1
alphas3 = np.array([0.15, 0.175, 0.1, 0.35, 0.175, 0.05])    #MASCHI >30
alphas4 = np.array([0.1, 0.06, 0.11, 0.185, 0.36, 0.185])    #FEMMINE >30

alphas_mean = (alphas2 + alphas3 + alphas4)/3


min_daily_users2 = 50
max_daily_users2 = 150
min_daily_users3 = 50
max_daily_users3 = 150
min_daily_users4 = 50
max_daily_users4 = 150

sold_items2 = np.array([2.9, 2.1, 0.1, 1.2, 2.5])      #Media di items venduti (la media viene sommata +1)
sold_items3 = np.array([0.8, 1.1, 1, 0.9, 0.6])
sold_items4 = np.array([0.4, 0.8, 0.8, 3.1, 1.5])
sold_items_mean = (sold_items2+sold_items3+sold_items4)/3


# DEFINIRE PROBABILITA DI PASSARE DA UN PRODOTTO A UN ALTRO PER OGNI CLASSE DI UTENTI

#np.array(alexa, quadro, cantinetta, pianta, pouf         articolo di partenza
#articolo che viene visualizzato
#alexa
#quadro
#cantinetta
#pianta
#pouf


#PERSONE <30
graph_probs2 = np.array([[0.0, 0.4, 0.4, 0.4, 0.4],
                        [0.3, 0.0, 0.3, 0.3, 0.3],
                        [0.2, 0.2, 0.0, 0.2, 0.2],
                        [0.3, 0.3, 0.3, 0.0, 0.3],
                        [0.6, 0.7, 0.8, 0.6, 0.0]]).T    #riga più alta perché prodotto che soddisfa di più la categoria

#MASCHI >30
graph_probs3 = np.array([[0, 0.3, 0.3, 0.3, 0.3],
                        [0.3, 0, 0.3, 0.3, 0.3],
                        [0.6, 0.7, 0, 0.6, 0.8],       #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.4, 0.4, 0.4, 0, 0.4],
                        [0.2, 0.2, 0.2, 0.2, 0]]).T

#FEMMINE >30
graph_probs4 = np.array([[0, 0.2, 0.2, 0.2, 0.2],
                        [0.3, 0, 0.3, 0.3, 0.3],
                        [0.3, 0.3, 0, 0.3, 0.3],
                        [0.8, 0.7, 0.6, 0.0, 0.6],      #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.4, 0.4, 0.4, 0.4, 0.4]]).T
 
graph_probs_mean = (graph_probs2 + graph_probs3 + graph_probs4)/3

l = 0.8

lambda_mat2 = np.array([[0.0, 0.0, 0.0, 1, l],
                        [1.0, 0.0, 1, 0.0, 1],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, l , 0.0, 0.0, 0.0],
                        [l , 1.0, l , l , 0.0]]).T 
