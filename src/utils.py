import itertools
import random

import networkx as nx
import numpy as np
import torch


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


def topological_overlap(A):
    numerator = (A[:, :-1, ...] * A[:, 1:, ...]).sum(-1)
    denominator = (A[:, :-1, ...].sum(-1) * A[:, 1:, ...].sum(-1)).sqrt() + 1e-6
    # shape (batch_size, time_len - 1, num_nodes)
    return (numerator / denominator)


def temporal_degree(A):
    numerator = A.sum(-1)
    denominator = 2 * (A.shape[-1] - 1)
    # shape (batch_size, time_len, num_nodes)
    return (numerator / denominator)


def index_fill(A, values, row_idx, col_idx):
    """Batched row and column fill"""
    assert values.shape[-1] == row_idx.shape[-1] == col_idx.shape[-1]
    dims = list(A.shape)
    batch_dims, num_nodes = dims[:-2], dims[-1]
    flat_batch_dims = np.prod(batch_dims)
    A, values = A.reshape(flat_batch_dims, num_nodes, num_nodes), values.reshape(flat_batch_dims, -1)
    row_idx, col_idx = row_idx.reshape(flat_batch_dims, -1), col_idx.reshape(flat_batch_dims, -1)
    A[:, row_idx, col_idx] = values[torch.arange(flat_batch_dims).unsqueeze(-1)]
    return A.reshape(*dims)


def get_adjacency_matrix(graph):
    A = nx.adjacency_matrix(graph).todense()
    return np.squeeze(np.asarray(A))
