import numpy as np
import scipy.sparse as sp


def set_diff_2d(A, B):
    
    nrows, ncols = A.shape
    
    dtype={"names":["f{}".format(i) for i in range(ncols)], "formats":ncols * [A.dtype]}
    out = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
    
    return out.view(A.dtype).reshape(-1, ncols)


def edge_sampler(adjacency_matrix, num_positive_edges=100, num_negative_edges=100):
    
    num_nodes = adjacency_matrix.shape[0]

    # lower triangle edges
    tril_idx = np.tril_indices(num_nodes, k=-1)
    tril_idx = np.array([tril_idx[0], tril_idx[1]]).T.astype(np.int32)

    # positive edges 
    row_idxs, col_idxs, weights = sp.find(adjacency_matrix)
    pos_edges = np.array([row_idxs, col_idxs]).T.astype(np.int32)
    num_pos = pos_edges.shape[0]

    # negative edges 
    neg_edges = set_diff_2d(tril_idx, pos_edges)
    num_neg = neg_edges.shape[0]

    pos_idx = np.random.choice(num_pos, num_positive_edges, replace=num_positive_edges > num_pos)
    sample_pos_edges = pos_edges[pos_idx, :]
    sample_pos_edge_weights = weights[pos_idx]

    neg_idx = np.random.choice(num_neg, num_negative_edges, replace=num_negative_edges > num_neg)
    sample_neg_edges = neg_edges[neg_idx, :]

    return sample_pos_edges, sample_pos_edge_weights, sample_neg_edges


