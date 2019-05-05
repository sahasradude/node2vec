import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from collections import defaultdict
from EvalUtils import EvalUtils

g = nx.read_weighted_edgelist("../datasets/redditdataset_75.txt", create_using=nx.DiGraph())
gtest= nx.read_weighted_edgelist("../datasets/redditdataset_test.txt", create_using=nx.DiGraph())
df1 = pd.read_csv('../datasets/redditdataset_75.txt', names = ['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None)
df2 = pd.read_csv('../datasets/redditdataset_test.txt', names = ['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None)

node2vec = Node2Vec(filename='../datasets/redditdataset_75.txt', is_temporal=True)  # Use temp_folder for big graphs
#model = KeyedVectors.load('somewhere',mmap = 'r')
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

all_connections = defaultdict(list)
for index, row in df1.iterrows():
        all_connections[row["v1"]].append((row["v2"]))


#all_connections_test = defaultdict(list)
# for index, row in df2.iterrows():
#     if row["v1"] not in all_connections:
#         all_connections_test[row["v1"]].append((row["v2"]))
#     else:
#         all_connections[row["v1"]].append((row["v2"]))


predicted = defaultdict(list)
count = 0
actual_links = set(gtest.edges)
print(actual_links)
preds = 0
actual_list = []
pred_list = []
for key in all_connections.keys():
    if g.out_degree(key) != 1:
        continue

    try:
        listobj = model.wv.most_similar(key)[:10]
        nodes = [elem[0] for elem in listobj]
        pred_edges = [(key, node) for node in nodes]
        pred_set = set(pred_edges)

        preds += len(actual_links.intersection(pred_set))
        pred_list.append(pred_edges)
        actual_list.append(actual_links)

    except KeyError:
        continue



print(EvalUtils.mapk(actual_list,pred_list,k=10))
print(preds)
