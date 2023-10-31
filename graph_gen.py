import numpy as np
import networkx as nx
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

def gen_Bconn_graphs(num_of_nodes, num_of_edges, B):
    '''Generate B connective subgraphs. Return the adjs of these subgraphs.'''

    # Generate strongly-connected directed graph
    G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=True)
    while nx.is_strongly_connected(G) == False:
        G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=True)

    # all links in G
    G_links = np.array([e for e in G.edges])

    num_avg_links = math.ceil(num_of_edges / B) # Just a notation. link # of each subgraph is the multiple of it (*2 or *3 ...).
    edge_perm = np.random.permutation(num_of_edges)
    edge_indexSet = np.concatenate((edge_perm, edge_perm)) # repeat twice (may change)

    adj_subG = np.zeros((B, num_of_nodes, num_of_nodes))

    for i in range(B):
        start_link_idx = i*num_avg_links
        end_link_idx = (i+2)*num_avg_links
        selected_links_idx = edge_indexSet[start_link_idx:end_link_idx]

        for idx in selected_links_idx:
            adj_subG[i, G_links[idx,0], G_links[idx,1]] = 1

    return adj_subG, G

def save_graphs(num_of_nodes, num_of_edges, B):
    '''Save the necessary data for the graphs.'''
    
    adj_subG, G = gen_Bconn_graphs(num_of_nodes, num_of_edges, B)
    save_dir = f'networks/N{num_of_nodes}E{num_of_edges}B{B}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    adj_G = nx.to_numpy_array(G)
    np.save(f'{save_dir}/adj_G.npy', adj_G)
    np.save(f'{save_dir}/adj_subG.npy', adj_subG)

    nx.draw_networkx(G, pos=nx.circular_layout(G), arrowsize=3, node_size=50, linewidths=0.2, width=0.2, with_labels=False)
    plt.savefig(f'{save_dir}/network.png')

    # transform the above adjs to weight matrices.
    W = np.zeros((B, num_of_nodes, num_of_nodes))
    for b in range(B):
        adj_selfloop = adj_subG[b] + np.identity(num_of_nodes)
        #inNeigh_arr = np.sum(adj_selfloop, axis=0)
        outNeigh_arr = np.sum(adj_selfloop, axis=1)
        for i in range(num_of_nodes):
            for j in range(num_of_nodes):
                if adj_selfloop[j, i] == 1.:
                    W[b, i, j] = 1 / outNeigh_arr[j]
    np.save(f'{save_dir}/W_subG.npy', W)