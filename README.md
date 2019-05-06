To test the link prediction task on the test set of the last 25% of links formed in the graph, run:

python3 test.py 


To run it on the larger reddit dataset, in test.py, change the dataset path and the training set path to:
"../datasets/redditdataset_75.txt", "../datasets/redditdataset_test.txt"

in the run function call.

The output of the function gives two lines; the number of links successfully predicted, 
and the Mean Average Precision @ 10 for the ranking of links returned for each node




