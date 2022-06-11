import numpy as np

np.random.seed(seed=456)


# DEFINIRE CONVERSION RATES PER OGNI CLASSE DI UTENTI

#CATEGORIA PRODOROTTI: ARTICOLI PER LA CASA
#Prodotti associati a categorie:
#MASCHI <30 --> ALEXA
#FEMMINE <30 --> POUF
#MASCHI >30 --> CANTINETTA PER IL VINI
#FEMMINE >30 --> PIANTA GRASSA
#QUADRO MEDIUM/HIGH PRICE PIACE A TUTTE LE CATEGORIE MA MENO DEI PREFERITI

#PRODOTTI CHE SODDISFANO CATEGORIE CON UN PARAMETRO IN COMUNE ED IL QUADRO SONO PREFRITI A PRODOTTI CHE NON LO FANNO
#esempio m<30 -->  pouf (<30) = cantinetta (maschio) > quadro > pianta

#MASCHI <30 ANNI
conversion_rates1 = np.array([[0.9, 0.75, 0.6, 0.3, 0.1],   #Alexa
                             [0.5, 0.4, 0.3, 0.1, 0.01],    #Quadro medium/high price
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Cantinetta per il vino
                             [0.4, 0,3, 0.2, 0.05, 0.01],   #Pianta grassa
                             [0.6, 0.45, 0.3, 0.1, 0.05]])  #Pouf
#FEMMINE <30 ANNI
conversion_rates2 = np.array([[0.6, 0.45, 0.3, 0.1, 0.05],  #Alexa
                             [0.5, 0.4, 0.3, 0.1, 0.01],    #Quadro medium/high price
                             [0.4, 0,3, 0.2, 0.05, 0.01],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Pianta grassa
                             [0.9, 0.75, 0.6, 0.3, 0.1]])   #Pouf
#MASCHI >30 ANNI
conversion_rates3 = np.array([[0.6, 0.45, 0.3, 0.1, 0.05],  #Alexa
                             [0.5, 0.3, 0.2, 0.1, 0.01],    #Quadro medium/high price
                             [0.9, 0.75, 0.6, 0.3, 0.1],    #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Pianta grassa
                             [0.4, 0,3, 0.2, 0.05, 0.01]])  #Pouf
#FEMMINE >30 ANNI
conversion_rates4 = np.array([[0.4, 0,3, 0.2, 0.05, 0.01],  #Alexa
                             [0.5, 0.3, 0.2, 0.1, 0.01],    #Quadro medium/high price
                             [0.6, 0.45, 0.3, 0.1, 0.05],   #Cantinetta per il vino
                             [0.9, 0.75, 0.6, 0.3, 0.1],    #Pianta grassa
                             [0.6, 0.45, 0.3, 0.1, 0.05]])  #Pouf

# DEFINIRE I PREZZI DI OGNI PRODOTTO
prices = np.array([[19.99, 24.99, 27.99, 33.99, 49.99],         #alexa
                  [169, 179, 189, 205, 215],                    #quadro
                  [229.99, 239.99, 249.99, 289.99, 329.99],     #cantinetta
                  [15.99, 19.99, 21.99, 22.99, 25.89],          #pianta
                  [64.99, 69.99, 72.99, 75.95, 79.99]])         #pouf

# DEFINIRE ALPHAS PER OGNI CLASSE DI UTENTI
# assunzione i maschi hanno meno probabilità di entrare in un sito di arredamento per la casa
# np.array([no sito, alexa, quadro, cantinetta, pianta, pouf])
alphas1 = np.array([0.15, 0.35, 0.1, 0.175, 0.05, 0.175])    #MASCHI <30              ELEMENTO 0 -> Non entra nel sito
alphas2 = np.array([0.1, 0.185, 0.11, 0.06, 0.185, 0.36])    #FEMMINE <30             LA SOMMA DELLE ALPHA E' 1
alphas3 = np.array([0.15, 0.175, 0.1, 0.35, 0.175, 0.05])    #MASCHI >30
alphas4 = np.array([0.1, 0.06, 0.11, 0.185, 0.36, 0.185])    #FEMMINE >30

min_daily_users1 = 100
max_daily_users1 = 500
min_daily_users2 = 100
max_daily_users2 = 500
min_daily_users3 = 100
max_daily_users3 = 500
min_daily_users4 = 100
max_daily_users4 = 500

max_sold_items1 = 5
max_sold_items2 = 5
max_sold_items3 = 5
max_sold_items4 = 5

# DEFINIRE PROBABILITA DI PASSARE DA UN PRODOTTO A UN ALTRO PER OGNI CLASSE DI UTENTI

#np.array(alexa, quadro, cantinetta, pianta, pouf         articolo di partenza
#articolo che viene visualizzato
#alexa
#quadro
#cantinetta
#pianta
#pouf

#0,2=peggioramento  0,3=stesse condizioni   0,4=miglioramento
#assunzione quadro = a prodotti che condividono una categoria 

#MASCHI <30
graph_probs1 = np.array([[0, 0.7, 0.6, 0.8, 0.6],   #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.2, 0, 0.3, 0.4, 0.3],
                        [0.2, 0.3, 0, 0.4, 0.3],
                        [0.1, 0.2, 0.2, 0, 0.2],
                        [0.2, 0.3, 0.3, 0.4, 0]])

#FEMMINE <30
graph_probs2 = np.array([[0, 0.3, 0.4, 0.3, 0.2],
                        [0.3, 0, 0.4, 0.3, 0.2],
                        [0.2, 0.2, 0, 0.2, 0.1],
                        [0.3, 0.3, 0.4, 0, 0.02],
                        [0.6, 0.7, 0.8, 0.6, 0]]) #riga più alta perché prodotto che soddisfa di più la categoria

#MASCHI >30
graph_probs3 = np.array([[0, 0.3, 0.2, 0.3, 0.4],
                        [0.3, 0, 0.2, 0.3, 0.4],
                        [0.6, 0.7, 0, 0.6, 0.8],    #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.3, 0.3, 0.2, 0, 0.4],
                        [0.2, 0.2, 0.1, 0.2, 0]])

#FEMMINIE >30
graph_probs4 = np.array([[0, 0.2, 0.2, 0.1, 0.2],
                        [0.4, 0, 0.3, 0.2, 0.03],
                        [0.4, 0.3, 0, 0.2, 0.3],
                        [0.8, 0.7, 0.6, 0, 0.06],  #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.4, 0.3, 0.3, 0.2, 0]])
