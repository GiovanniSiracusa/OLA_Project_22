import numpy as np
from copy import copy


def simulate_episode(init_prob_matrix, n_steps_max, initial_active_node, l=0.8):
    prob_matrix = init_prob_matrix.copy()
    history = np.array([initial_active_node])
    previous_all = np.zeros(n_nodes, dtype=np.int8)-2
    previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

    active_nodes = np.array([initial_active_node])  # array di active nodes esempio: se active_nodes sono 2 e 3 allora sarà [[0,1,0,0,0], [0,0,1,0,0]]
    newly_active_nodes = active_nodes
    t = 0
    while (t < n_steps_max and len(active_nodes) > 0): #np.sum(newly_active_nodes) > 0):
        active_node = active_nodes[0].copy()  # Prendi il primo active node in coda
        #print("Active Node:", active_node)
        active_nodes = active_nodes[1:]
        idx_active = np.argwhere(active_node).reshape(-1)

        p = (prob_matrix.T * active_node).T
        maxs = np.argsort(p[idx_active]).reshape(-1)

        # for idx in range(len(p[idx_active].reshape(-1))):
        #     if maxs[-1] != idx and maxs[-2] != idx:
        #         p[idx_active, idx] = 0
        #     if maxs[-2] == idx:
        #         p[idx_active, idx] = p[idx_active, idx] * l

        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])  # AGGIUNGERE I RESERVATION PRICE E LAMBDA
        # prob_matrix = prob_matrix * ((p != 0) == activated_edges)

        prob_matrix[:, idx_active] = 0
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_node)
        #print("Newly active:", newly_active_nodes)

        # Split newly active nodes
        for idx, val in enumerate(newly_active_nodes):
            if val == 1:
                prob_matrix[:, idx] = 0
                a = np.zeros(5)
                a[idx] = 1
                active_nodes = np.concatenate((active_nodes, [a]), axis=0)
                previous_all[idx] = idx_active

        #active_nodes = np.unique(active_nodes, axis=0)
        # print(active_nodes, end='\n\n')
        history = np.concatenate((history, [active_node]), axis=0)
        t += 1
    history = history[1:]
    # print(prob_matrix)
    previous = np.array([], dtype=np.int8)
    for e in history:
        previous = np.append(previous, previous_all[np.argwhere(e).reshape(-1)])
    return history, previous

def simulate_episode2(init_prob_matrix, initial_active_node, l=0.8):
    if all(initial_active_node == 0):
        return [], []
    prob_matrix = init_prob_matrix.copy()
    np.fill_diagonal(prob_matrix, 0)
    history = np.empty((0, n_nodes))
    active_nodes_list = np.array([initial_active_node])
    previous_all = np.zeros(n_nodes, dtype=np.int8)-2
    previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

    t = 0
    while (len(active_nodes_list) > 0):
        active_node = active_nodes_list[0].copy()
        active_nodes_list = active_nodes_list[1:]
        idx_active = np.argwhere(active_node).reshape(-1)
        #print("Active node ", active_node, end='\n')
        
        res_price = np.random.uniform(0, 1) # Cambiamo distribuzione
        
        if res_price < 0.7:
            p = (prob_matrix.T * active_node).T
            rnd = np.random.choice(np.where(np.arange(n_nodes) != idx_active)[0], 2, replace=False)
            #print("Possible choice: ", rnd)
            for i in range(n_nodes):
                # Multiply by lambda the secondary product in the second slot
                if i == rnd[0]:
                    pass
                elif i ==  rnd[1]:
                    p[idx_active,i] = p[idx_active, i] * l
                else:
                    p[idx_active, i] = 0

            #print(p)
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            #print("Activated edges: ", activated_edges)
            prob_matrix[:, idx_active] = 0
            
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_node)
            #print("Newly active nodes: ", newly_active_nodes)
            # Split newly active nodes

            for idx in rnd:
                if newly_active_nodes[idx] == 1:
                    prob_matrix[:, idx] = 0
                    a = np.zeros(5)
                    a[idx] = 1
                    active_nodes_list = np.concatenate((active_nodes_list, [a]), axis=0)
                    previous_all[idx] = idx_active
            #print(active_nodes_list)
        history = np.concatenate((history, [active_node]), axis=0)

    previous = np.array([], dtype=np.int8)
    for e in history:
        previous = np.append(previous, previous_all[np.argwhere(e).reshape(-1)])
    return history, previous

# def estimate_probabilities(dataset, node_index, n_nodes):
#     estimated_prob = np.ones(n_nodes) * 1.0 / (n_nodes - 1)
#     credits = np.zeros(n_nodes)
#     occur_v_active = np.zeros(n_nodes)
#     n_episodes = len(dataset)
#     for episode in dataset:
#         idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
#         if len(idx_w_active) > 0 and idx_w_active > 0:
#             active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
#             credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
#         for v in range(0, n_nodes):
#             if (v != node_index):
#                 idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
#                 if len(idx_v_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
#                     occur_v_active[v] += 1
#     estimated_prob = credits / occur_v_active
#     estimated_prob = np.nan_to_num(estimated_prob)
#     return estimated_prob

def estimate_probabilities(dataset, previous, n_nodes):
    credits = np.zeros((n_nodes, n_nodes))
    active = np.zeros(n_nodes)

    for episode, prev in zip(dataset, previous):
        for e, p in zip(episode, prev):
            idx = np.argwhere(e).reshape(-1)
            active[idx] += 1
            if p >= 0:
                credits[p, idx] += 1
            else:
                credits[idx, idx] += 0
    print(credits)
    print(active)
    #print(np.sum(credits, axis=0))
    for i in range(n_nodes):
        for j in range(n_nodes):
            credits[i,j] = credits[i,j] / active[i]
    #print(credits.T)
    return credits
    
n_nodes = 5
n_episodes = 1000
prob_matrix = np.random.uniform(0, 0.3, (n_nodes, n_nodes))
np.fill_diagonal(prob_matrix, 0)

lmbd = 0.8

graph = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    rnd = np.random.choice(np.where(np.arange(n_nodes) != i)[0], 2, replace=False)
    graph[i][rnd[0]] = 1
    graph[i][rnd[1]] = lmbd

nodes = np.array(range(n_nodes + 1))
alphas = np.random.dirichlet(np.ones(6), size=1).reshape(-1)
print(alphas)
# print(prob_matrix)
# print(graph)
# print(prob_matrix*graph)
# print("*********************")
dataset = []
previous = []


for e in range(0, n_episodes):
    initial_node = np.random.choice(nodes, 1, p=alphas)
    initial_active_node = np.zeros(n_nodes + 1)
    initial_active_node[initial_node] = 1
    initial_active_node = initial_active_node[1:]
    history, prev = simulate_episode2(init_prob_matrix=prob_matrix, initial_active_node=initial_active_node)
    dataset.append(history)
    previous.append(prev)
    #print(previous)
    #print(dataset)

#estimated_prob = estimate_probabilities(dataset=dataset, node_index=1, n_nodes=n_nodes)
estimated_prob = estimate_probabilities(dataset=dataset,previous=previous, n_nodes=n_nodes)
print(f"{estimated_prob}")

print("True P Matrix: ")
print(prob_matrix)
print()
print((prob_matrix-estimated_prob)/prob_matrix)