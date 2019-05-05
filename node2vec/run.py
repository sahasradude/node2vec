from node2vec import Node2Vec


filename = '../datasets/collegedataset_75.txt'
n2v = Node2Vec(filename, is_temporal=True, model_file="model.n2v")
# n2v = Node2Vec(filename, is_temporal=True)
i = j = 0
for node in n2v.cold_starts:
    print("--------------------------------------------------")
    print("Node=", node)
    fof = n2v.find_fof(node)
    print("Friend of friend=", fof)
    if node == fof[0]:
        print(node, fof[0])
        i+=1
    else:
        j+=1

print(i, j)

# print(len(n2v.cold_starts))
# print(len(g.nodes))

test_filename = '../datasets/collegedataset_test.txt'

