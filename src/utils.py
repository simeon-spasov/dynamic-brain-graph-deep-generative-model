import itertools
import random


def sample_pos_neg_edges(graph, num_samples=1):
    # positive edges from networkx graph
    pos_edges = set(graph.edges())
    # all possible edge combinations
    all_edges = set(itertools.combinations(graph.nodes(), 2))
    # find all possible negative edges
    all_neg_edges = all_edges - pos_edges

    num_edges = len(pos_edges)

    neg_edges = []
    pos_edges = []

    # sample negative edges
    neg_edges += [random.sample(list(all_neg_edges), num_edges) for _ in range(num_samples)]

    pos_edges += [list(graph.edges()) for _ in range(num_samples)]

    return list(itertools.chain.from_iterable(pos_edges)), \
           list(itertools.chain.from_iterable(neg_edges))
