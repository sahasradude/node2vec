import math
from collections import defaultdict
import numpy as np
import gensim, os
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import networkx as nx
import random
from random import sample
from random import shuffle
from numpy.random import choice


class FriendOfFriend:
    """
    Finds the friend of friend temporal embedding for a directed temporal graph
    """

    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'timestamp'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'


    def __init__(self, filename, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='timestamp',
                 workers=1, sampling_strategy=None, quiet=False, temp_folder=None, is_temporal=False, model_file=None):

        self.has_cold_started = set()
        self.cold_started_with = defaultdict(list)

        self.filename = filename

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
        self.k = 5
        self.cold_starts = set()
        self.model = None
        self.embeddings = None
        self.model_file = model_file
        self.time_dict = defaultdict(list)
        self.timestamps = dict()
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
        initialize graph, make cold_start dict, timestamp dict, make temporal weighting for timestamps
        :return:
        """

        df1 = pd.read_csv(self.filename, names=['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None, dtype=str)

        # all the connections
        for index, row in df1.iterrows():
            self.time_dict[str(row["v1"])].append(float(row["timestamp"]))



        for index, row in df1.iterrows():
            l = self.time_dict[str(row["v1"])]
            ts_last = l[len(l) - 1]
            ts = float(row["timestamp"])
            row["timestamp"] = 1.0 / (math.e ** (self.k * (abs(ts - ts_last) / ts_last) ))

            if row["v1"] not in self.has_cold_started:
                self.has_cold_started.add(row["v1"])
                self.cold_started_with[str(row["v2"])].append((str(row["v1"]), row["timestamp"]))


        self.graph = self.read_graph(df1)

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
            delayed(self.parallel_generate_walks)(self.d_graph,
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
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the FriendOfFriend 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)

    def read_graph(self, df):

        try:
            g = nx.from_pandas_edgelist(df, "v1", "v2", edge_attr=True, create_using=nx.DiGraph())
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
        edges = list(g.edges(node, data=True))
        if len(edges) == 0:
            return node
        s = sum([edge[2]["timestamp"] for edge in edges])
        if s == 0:
            prob_list = [1/len(edges) for edge in edges]

        else:
            prob_list = [edge[2]["timestamp"] / s for edge in edges]

        e = [edge[1] for edge in edges]
        friend = choice(e, 1, p=prob_list)[0]

        fof_list = list(g.edges(friend))

        fof_set = set(elem[0] for elem in fof_list)

        fof_set.difference_update(set(node))
        if len(fof_set) == 0:
            return [node]

        return sample(fof_set, 1)



    def parallel_generate_walks(self, d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                                num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
                                first_travel_key=None, quiet=False):


        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        walks = list()
        if not quiet:
            pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):

            # Update progress bar
            if not quiet:
                pbar.update(1)

            # Shuffle the nodes
            shuffled_nodes = list(d_graph.keys())
            shuffle(shuffled_nodes)
            for source in shuffled_nodes:
                # TODO: Remove and uncomment below after testing
                prob = 1/(1+self.graph.out_degree(source))

                rand = random.uniform(0,1)
                if rand > prob:
                    try:
                        walk = self.single_node_random_walk(source, sampling_strategy, num_walks_key, n_walk, walk_length_key,
                                                            global_walk_length, d_graph, neighbors_key, first_travel_key, probabilities_key, cold_start=False, real_start_node=None)
                        walks.append(walk)
                    except:
                        rand = 0

                elif rand <= prob:

                    new_source_list = self.find_fof(source)
                    if type(new_source_list) is not list:
                        new_source_list = [new_source_list]
                    for elem in new_source_list:
                        try:
                            walk = self.single_node_random_walk(elem, sampling_strategy, num_walks_key, n_walk, walk_length_key,
                                                        global_walk_length, d_graph, neighbors_key, first_travel_key, probabilities_key, cold_start=True, real_start_node=source)
                            walks.append(walk)
                        except:
                            pass

        if not quiet:
            pbar.close()

        return walks


    def single_node_random_walk(self, source, sampling_strategy, num_walks_key, n_walk, walk_length_key,
                                global_walk_length, d_graph, neighbors_key, first_travel_key, probabilities_key, cold_start=False, real_start_node=None):
        # Start a random walk at the source node
        # Skip nodes with specific num_walks
        if source in sampling_strategy and \
            num_walks_key in sampling_strategy[source] and \
            sampling_strategy[source][num_walks_key] <= n_walk:
            return

        # Start walk
        if cold_start is False:
            walk = [source]
        else:
            walk = [real_start_node]

        # Calculate walk length
        if source in sampling_strategy:
            walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
        else:
            walk_length = global_walk_length

        # Perform walk
        while len(walk) < walk_length:

            walk_options = d_graph[walk[-1]].get(neighbors_key, None)

            # Skip dead end nodes
            if not walk_options:
                break

            if len(walk) == 1:  # For the first step
                probabilities = d_graph[walk[-1]][first_travel_key]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
            else:
                probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

            walk.append(walk_to)

        walk = list(map(str, walk))  # Convert all to strings
        return walk

