import networkx as nx
from random import sample
from parallel import *
from node2vec import Node2Vec
import os

class Utils:

    def read_graph(self, filename):
        g = nx.read_weighted_edgelist(filename, create_using=nx.DiGraph())
        return g

    def find_fof(self, node, n2v_obj):
        """
        finds the "friend of friend node" for a given cold_start node
        :param node: cold start node
        :param n2v_obj: the node2vec object in question
        :return: a sample of "friend of friend nodes"
        """
        #TODO:find which performs best, single random sample vs average of many random samples
        g = n2v_obj.graph
        edge = list(g.edges(node, data=True))
        friend_edge = edge
        friend = friend_edge[0][1]

        print("Friend =", friend)

        fof_list = list(g.edges(friend, data=True))
        fof_set = set(elem[1] for elem in fof_list)
        intersection = fof_set.intersection(n2v_obj.cold_starts)
        try:
            intersection.remove(node)
        except KeyError:
            pass

        if node in intersection:
            print("WTF")

        if len(intersection) == 0:
            print(len(list(g.edges(friend))))
            return [node]

        return sample(intersection, 1)
        # for fof_edge in intersection:
        #     if fof_edge[1] != node:
        #         timestamp = fof_edge[2]["weight"]











u = Utils()
dir = os.getcwd()
g = u.read_graph('../datasets/collegedataset_75.txt')
n2v = Node2Vec(g, is_temporal=True, model_file="model.n2v")
# n2v = Node2Vec(g, is_temporal=True)
i = j = 0
for node in n2v.cold_starts:
    print("--------------------------------------------------")
    print("Node=", node)
    fof = u.find_fof(node, n2v)
    print("Friend of friend=", fof)
    if node == fof[0]:
        print(node, fof[0])
        i+=1
    else:
        j+=1

print(i, j)

# print(len(n2v.cold_starts))
# print(len(g.nodes))



