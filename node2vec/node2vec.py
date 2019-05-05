from collections import defaultdict
import numpy as np
import gensim, os
from gensim.models import KeyedVectors
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
from parallel import parallel_generate_walks
import pandas as pd
import networkx as nx
from random import sample


class Node2Vec:
    """
    Given the entire history of the graph, see which nodes are cold starts right now, and which ones are not, and
    generate either our embedding, or node2vec
    """
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'


    def __init__(self, filename, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, temp_folder=None, is_temporal=False, model_file=None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :type graph: Networkx Graph
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        :type temp_folder: str
        """
        self.has_cold_started = set()
        self.cold_started_with = defaultdict(list)

        self.filename = filename

        self.graph = self.read_graph(filename)
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)
        self.is_temporal = is_temporal

        self.cold_starts = set()
        self.model = None
        self.embeddings = None
        self.model_file = model_file

        if model_file is not None:
            if not os.path.exists(model_file):
                print("Model file path does not exist, please check model file path and try again")
                return
            self.embedding_file = model_file+".emb"


        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        self._initialize_structures()

    def _initialize_structures(self):
        """
        initialize the forward and reverse dictionaries for the current graph
        :return:

        """

        df1 = pd.read_csv(self.filename, names=['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None)

        # all the connections
        for index, row in df1.iterrows():
            if row["v1"] not in self.has_cold_started:
                self.has_cold_started.add(row["v1"])
                self.cold_started_with[str(row["v2"])].append((str(row["v1"]), row["timestamp"]))

        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 1:
                self.cold_starts.add(node)

        if self.model_file is None:
            self._precompute_probabilities()
            self.walks = self._generate_walks()
            self.model = self.fit(window=10, min_count=1, batch_words=4)
            self.model_file = "model.n2v"
            self.embedding_file = "model.n2v.emb"
            self.model.save(self.model_file)
            self.model.wv.save(self.embedding_file)

        else:
            self.model = KeyedVectors.load(self.model_file, mmap='r')
            self.embeddings = KeyedVectors.load(self.embedding_file, mmap='r')




    def maintain_embeddings(self):
        """
        figures out when a node stops being cold start, and recomputes its embedding using node2vec
        :return:
        """

    def _precompute_probabilities(self):

        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks


    def fit(self, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)

    def read_graph(self, filename):

        try:
            g = nx.read_weighted_edgelist(filename, create_using=nx.DiGraph())
            return g
        except FileNotFoundError:
            print('This file does not exist or is corrupted')
            return


    def find_fof(self, node):
        """
        finds the "friend of friend node" for a given cold_start node
        :param node: cold start node
        :return: a sample of "friend of friend nodes"
        """
        #TODO:find which performs best, single random sample vs average of many random samples
        g = self.graph
        edge = list(g.edges(node, data=True))
        friend_edge = edge
        friend = friend_edge[0][1]

        print("Friend =", friend)

        fof_list = self.cold_started_with[friend]

        print(fof_list)
        print(node)

        fof_set = set(elem[0] for elem in fof_list)

        fof_set.remove(node)

        if len(fof_set) == 0:
            return [node]

        return sample(fof_set, 1)
        # for fof_edge in intersection:
        #     if fof_edge[1] != node:
        #         timestamp = fof_edge[2]["weight"]

