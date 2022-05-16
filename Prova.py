import numpy as np
from numpy.random import choice


def simulate_episode(init_prob_matrix, n_steps_max):        # PER CALCOLARE UN CASO. PARTE DALLA MATRICE DI PROBABILITA E SIMULA LA PROPAGAZIONE NEI NODI. ESERCITAZIONE 1
                                                            # SE ATTIVO ALPHA 0, SI PASSA MATRICE A 0




alphas = np.random.dirichlet(np.ones(6), size=1).reshape(-1)  # Prende 6 samples la cui somma è 1
print("Alphas :")
print(alphas)
n_nodes = 5
nodes = np.array([0, 1, 2, 3, 4, 5])
prob_matrix = np.random.uniform(0.0, 0.3, (n_nodes, n_nodes))
np.fill_diagonal(prob_matrix, 0)
print("Prob Matrix :")
print(prob_matrix)
initial_node = choice(nodes, 1, p=alphas)  # Come gestire il caso in cui venga scelto alpha_0?
print("Initial node:")
print(initial_node)
clicked_nodes = np.array([initial_node])
print("clicked_nodes")
print(clicked_nodes)
active_node = initial_node
print("active node")
print(active_node)

