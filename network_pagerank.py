import networkx as nx
import numpy as np
from solver import Solver

G = nx.karate_club_graph()
adj = np.array(nx.adj_matrix(G).todense())
n_nodes = adj.shape[0]
row_norm = np.array(np.sum(adj, axis=1).reshape(n_nodes))
adj_norm = adj / row_norm[:, None]
# print(np.sum(adj_norm, axis=1))
d = 0.85
A = np.array(np.identity(n_nodes) - d * adj_norm.transpose())
b = (1 - d) * np.ones((n_nodes, 1)) / n_nodes
solver = Solver(A, b)
# solver.conj_grad()   # adj is not symmetric positive definite
solver.sor()

ans = nx.pagerank(G)
print('ans', ans)
