import networkx as nx
from node2vec import Node2Vec
import os
class Utils:

    def read_graph(self, filename):
        g = nx.read_weighted_edgelist(filename, create_using=nx.DiGraph())
        return g



u = Utils()
dir = os.getcwd()
g = u.read_graph('../datasets/collegedataset.txt')
n2v = Node2Vec(g, is_temporal=True, model_file="model.n2v")

print(len(n2v.cold_starts))
print(len(g.nodes))

