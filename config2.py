import numpy as np

#np.random.seed(seed=1)


# DEFINIRE CONVERSION RATES PER OGNI CLASSE DI UTENTI

#CATEGORIA PRODOROTTI: ARTICOLI PER LA CASA
#Prodotti associati a categorie:
#MASCHI <35 --> ALEXA
#FEMMINE <35 --> POUF
#MASCHI >35 --> CANTINETTA PER IL VINI
#FEMMINE >35 --> PIANTA GRASSA
#QUADRO MEDIUM/HIGH PRICE PIACE A TUTTE LE CATEGORIE MA MENO DEI PREFERITI

#PRODOTTI CHE SODDISFANO CATEGORIE CON UN PARAMETRO IN COMUNE ED IL QUADRO SONO PREFRITI A PRODOTTI CHE NON LO FANNO
#esempio m<35 -->  pouf (<35) = cantinetta (maschio) > quadro > pianta

#MASCHI <35 ANNI
#conversion_rates1 = np.array([[0.9, 0.75, 0.6, 0.3],  #Alexa
#                             [0.5, 0.4, 0.3, 0.1],    #Quadro medium/high price
#                             [0.6, 0.45, 0.3, 0.1],   #Cantinetta per il vino
#                             [0.4, 0.3, 0.2, 0.05],   #Pianta grassa
#                             [0.6, 0.45, 0.3, 0.1]])  #Pouf

#riduzione dei conversion rate del 30% dovuta dalla crisi economica e quindi una tendenza a non acqiostare più alta su tutte le classi
#FEMMINE <35 ANNI
conversion_rates2 = np.array([[0.6, 0.45, 0.3, 0.1],  #Alexa
                             [0.5, 0.4, 0.3, 0.1],    #Quadro medium/high price
                             [0.4, 0.3, 0.2, 0.05],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1],   #Pianta grassa
                             [0.9, 0.75, 0.6, 0.3]])  #Pouf
conversion_rates2 = 0.7*conversion_rates2
#MASCHI >35 ANNI
conversion_rates3 = np.array([[0.6, 0.45, 0.3, 0.1],  #Alexa
                             [0.5, 0.3, 0.2, 0.1],    #Quadro medium/high price
                             [0.9, 0.75, 0.6, 0.3],   #Cantinetta per il vino
                             [0.6, 0.45, 0.3, 0.1],   #Pianta grassa
                             [0.4, 0.3, 0.2, 0.05]])  #Pouf
conversion_rates3 = 0.7*conversion_rates3
#FEMMINE >35 ANNI
conversion_rates4 = np.array([[0.4, 0.3, 0.2, 0.05],  #Alexa
                             [0.5, 0.3, 0.2, 0.1],    #Quadro medium/high price
                             [0.6, 0.45, 0.3, 0.1],   #Cantinetta per il vino
                             [0.9, 0.75, 0.6, 0.3],    #Pianta grassa
                             [0.6, 0.45, 0.3, 0.1]])  #Pouf
conversion_rates4 = 0.7*conversion_rates4

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
#alphas1 = np.array([0.15, 0.35, 0.1, 0.175, 0.05, 0.175])    #MASCHI <35              ELEMENTO 0 -> Non entra nel sito
alphas2 = np.array([0.1, 0.185, 0.11, 0.06, 0.185, 0.36])    #FEMMINE <35             LA SOMMA DELLE ALPHA E' 1
alphas3 = np.array([0.15, 0.175, 0.1, 0.35, 0.175, 0.05])    #MASCHI >35
alphas4 = np.array([0.1, 0.06, 0.11, 0.185, 0.36, 0.185])    #FEMMINE >35

alphas_mean = (alphas2 + alphas3 + alphas4)/3

#min_daily_users1 = 100
#max_daily_users1 = 500
min_daily_users2 = 50
max_daily_users2 = 150
min_daily_users3 = 50
max_daily_users3 = 150
min_daily_users4 = 50
max_daily_users4 = 150

#max_sold_items1 = 5
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

#0,2=peggioramento  0,3=stesse condizioni   0,4=miglioramento
#assunzione quadro = a prodotti che condividono una categoria

#MASCHI <35
#graph_probs1 = np.array([[0, 0.7, 0.6, 0.8, 0.6],     #riga più alta perché prodotto che soddisfa di più la categoria
#                        [0.2, 0, 0.3, 0.4, 0.3],
#                        [0.2, 0.3, 0, 0.4, 0.3],
#                        [0.1, 0.2, 0.2, 0, 0.2],
#                        [0.2, 0.3, 0.3, 0.4, 0]]).T

#FEMMINE <35
graph_probs2 = np.array([[0.0, 0.0, 0.0, 0.3, 0.2],
                        [0.3, 0.0, 0.4, 0.0, 0.2],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.3, 0.0, 0.0, 0.0],
                        [0.6, 0.7, 0.8, 0.6, 0.0]]).T    #riga più alta perché prodotto che soddisfa di più la categoria

#MASCHI >35
graph_probs3 = np.array([[0, 0.0, 0.2, 0.3, 0.4],
                        [0.0, 0, 0.2, 0.0, 0.0],
                        [0.6, 0.7, 0, 0.6, 0.8],       #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.0, 0.3, 0.0, 0, 0.0],
                        [0.2, 0.0, 0.0, 0.0, 0]]).T

#FEMMINIE >35
graph_probs4 = np.array([[0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0, 0.3, 0.2, 0.0],
                        [0.4, 0.3, 0, 0.2, 0.3],
                        [0.8, 0.7, 0.6, 0.0, 0.6],      #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.0, 0.0, 0.0, 0.0, 0.0]]).T

graph_probs_mean = (graph_probs2 + graph_probs3 + graph_probs4)/3


l = 0.8

lambda_mat2 = np.array([[0.0, 0.0, 0.0, 1, l],
                        [1.0, 0.0, 1, 0.0, 1],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, l , 0.0, 0.0, 0.0],
                        [l , 1.0, l , l , 0.0]]).T 

lambda_mat3 = np.array([[0, 0.0, l, l, 1],
                        [0.0, 0, 1, 0.0, 0.0],
                        [1.0, 1, 0, 1, l],       #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.0, l, 0.0, 0, 0.0],
                        [l, 0.0, 0.0, 0.0, 0]]).T

lambda_mat4 = np.array([[0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0, 1, 1, 0.0],
                        [1, l, 0, l, 1],
                        [l, 1, l, 0.0, l],      #riga più alta perché prodotto che soddisfa di più la categoria
                        [0.0, 0.0, 0.0, 0.0, 0.0]]).T