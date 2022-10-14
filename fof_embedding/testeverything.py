import networkx as nx
import pandas as pd
from FriendofFriend import FriendOfFriend
from gensim.models import KeyedVectors
from collections import defaultdict
from EvalUtils import EvalUtils

#g = nx.read_weighted_edgelist("../datasets/redditdataset_75.txt", create_using=nx.DiGraph())
missed = 0
df = pd.read_csv('../datasets/collegedataset.txt', names = ['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None, dtype = str)
sumpred = 0
summap = 0
avg_f1 = 0
for i in range(1,5):

    try:



        dftrain = df[:int(i*len(df)/5)]
        dftest =  df[int(i*len(df)/5 + 1):]



        graph = nx.from_pandas_edgelist(dftrain,source='v1',
                                           target='v2',edge_attr='timestamp',
                                           create_using=nx.DiGraph())

        testgraph = nx.from_pandas_edgelist(dftest, source='v1',
                                        target='v2', edge_attr='timestamp',
                                        create_using=nx.DiGraph())

        actual = set((edge[0], edge[1]) for edge in testgraph.edges())
        nodes = graph.nodes

        node2vec = FriendOfFriend(df = dftrain, num_walks=10, walk_length=8)


        model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
        predset = set()
        for node in nodes:
            listobj = model.wv.most_similar(node)[0][0]
            pred_edge = (node, listobj)
            predset.add(pred_edge)

        precision = len(set(actual).intersection(predset)) / len(predset)
        recall = len(set(actual).intersection(predset)) / len(actual)

        f1_score = (2 * precision * recall) / (precision + recall)
        print("p",precision)
        print("r",recall)
        print ("f1",f1_score)

        avg_f1 += f1_score

    except KeyError:
        missed+=1
        continue

print("avg f1:", avg_f1 / 5)

