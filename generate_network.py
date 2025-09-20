import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def gener_net_erdos(N, p):
    rg=nx.erdos_renyi_graph(N, p)
    ps=nx.shell_layout(rg)
    nx.draw(rg, ps, with_labels=True, node_size=200)
    # plt.savefig('./test1.png')
    adj_matrix=nx.adjacency_matrix(rg)
    return adj_matrix.todense()


def gener_net_regular(d, N):
    rg=nx.random_regular_graph(d, N)
    ps=nx.shell_layout(rg)
    nx.draw(rg, ps, with_labels=True, node_size=200)
    plt.savefig('./test1.png')
    adj_matrix=nx.adjacency_matrix(rg)
    return adj_matrix.todense()

'''
N = [4, 6, 8, 10, 12, 14, 16]
p = 0.5
for n in N:
    adj_matrix = gener_net_erdos(n, p)
    while np.any(np.sum(adj_matrix, axis=1)==0):
        adj_matrix = gener_net_erdos(n, p)
    np.savetxt('./Node/top/adj_%d'%n, adj_matrix, fmt='%i')

adj_matrix=np.matrix([[0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
                    , [1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
                    , [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
                    , [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                    , [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
                    , [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
                    , [0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
                    , [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
                    , [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]])

adj_matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

graph = nx.from_numpy_matrix(adj_matrix)
nx.draw(graph,pos = nx.spring_layout(graph), node_color = 'y',edge_color = 'k',with_labels = True, width=0.5, font_size=15, node_size =350)
plt.show()
'''