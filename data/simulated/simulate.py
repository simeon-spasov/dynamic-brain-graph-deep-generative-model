# import pathlib
# import random
import itertools
import numpy as np
import scipy.sparse as sp


def shuffle_split(list_, num_splits):

    np.random.shuffle(list_)
    quotient, remainder = divmod(len(list_), num_splits)

    list_splits = []

    for split in range(num_splits):
        from_idx = split * quotient + min(split, remainder)
        to_idx = (split + 1) * quotient + min(split + 1, remainder)
        
        list_section = list_[from_idx:to_idx]
        list_section.sort()
        
        list_splits += [list_section]
    
    return list_splits


def cartesian_product(*arrays):
    """
    Compute the cartesian product of multiple arrays
    """
    N = len(arrays)
    return np.transpose(np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)).reshape(-1, N)
    

def group_sizes(num_samples, num_groups):
        
    # split the number of samples equally between groups
    quotient, remainder = divmod(num_samples, num_groups)
    sizes = [quotient] * num_groups
    sizes[np.random.choice(num_groups, 1)[0]] += remainder

    return sizes


def node_community_assignment(community_sizes):
    
    community_nodes = {}
    
    idx = 0
    for community, size in enumerate(community_sizes):
        
        community_nodes[community] = list(range(idx, idx + size))
        idx += size

    return community_nodes


def node_feature_distribution(num_features, num_communities, mean=1., var=0.5):

    mean = np.array([mean, -mean] * num_features)
    var = np.array([var, var + 0.25] * num_features)

    perm = list(set(itertools.permutations(range(len(mean)), num_features)))
    np.random.shuffle(perm)
    
    mean = [mean[np.argsort(p)] for p in perm][:num_communities]
    var = [var[np.argsort(p)] for p in perm][:num_communities]
    
    return mean, var


def sample_edge_weights(node_community, intra_weight_dist, inter_weight_dist):

    num_communities = len(node_community)
    num_nodes = len(np.unique(np.concatenate(list(node_community.values()), axis=-1)))

    W = np.random.normal(**intra_weight_dist, size=(num_nodes, num_nodes))

    for c in range(num_communities):

        cartesian_prod = cartesian_product(node_community[c], node_community[c])

        if len(cartesian_prod) > 0:

            W[cartesian_prod[:, 0], cartesian_prod[:, 1]] = np.random.normal(size=len(cartesian_prod), **inter_weight_dist[c])

    W = ((W - np.min(W)) / np.ptp(W)) + 1e-6 

    W_diag = list(W.diagonal())
    W *= np.tri(*W.shape, k=-1).astype(W.dtype)
    W += W.T 
    np.fill_diagonal(W, W_diag)

    return W


def sample_node_features(node_community, mean, var):

    nodes, communities = node_community.nonzero()
    nodes, counts = np.unique(nodes, return_counts=True)
    
    num_nodes = len(nodes)
    num_features = len(mean[0])

    node_features = np.zeros((num_nodes, num_features))

    for node, community in zip(nodes, communities):
    
        node_features[node] += (np.random.standard_normal(num_features) * var[community] + mean[community]) / counts[node]

    return node_features


def overlap_community_nodes(node_community, community_from, community_to, prop_overlap):
    
    end_node = max(node_community[community_from])
    overlap_nodes = list(range(end_node - int(end_node * prop_overlap), end_node))
    node_community[community_to] += overlap_nodes

    return node_community


def node_community_to_onehot(node_community):

    num_communities = len(node_community)
    num_nodes = max(node_community[num_communities - 1]) + 1

    node_community_one_hot = np.zeros((num_nodes, num_communities))
    for community in range(num_communities):
        for node in node_community[community]:
            node_community_one_hot[node, community] = 1.

    return node_community_one_hot


def sample_adjacency(prob_edge, self_loops=False, sparse=False):

    assert np.alltrue(prob_edge == prob_edge.T)
    assert np.alltrue(np.greater_equal(prob_edge, 0.) == np.less_equal(prob_edge, 1.))
    
    prob_edge = 1 - np.exp(-prob_edge)
    adjacency = np.random.binomial(1, p=prob_edge).astype(np.int8)

    # upper triangle to zero
    adjacency *= np.tri(*adjacency.shape, k=-1).astype(adjacency.dtype)
    adjacency += adjacency.T 

    if self_loops:
        np.fill_diagonal(adjacency, 1.)
    else:
        np.fill_diagonal(adjacency, 0.)

    assert np.alltrue(adjacency == adjacency.T)

    if sparse:
        # sparse matrix format
        adjacency = sp.csr_matrix(adjacency)

    return adjacency


# def simulate_community_migration_dataset():
#     num_samples = 100
#     num_nodes = 100
#     num_communities = 4 
#     time_len = 10
#     community_sizes = [num_nodes // num_communities] * num_communities

#     num_classes = 2
#     num_samples_per_class = [num_samples // num_classes] * num_classes
#     intra_edge_prob = [0.001, 0.001] 
#     inter_edge_prob = [0.2, 0.2]
    
#     migrate_from_community = [0, 2]
#     migrate_to_community = [1, 3]

#     data_dir = pathlib.Path("./migration")
#     data_dir.mkdir(exist_ok=True)

#     counter = 0
    
#     for label in range(num_classes):

#         community_edge_probability = np.ones((num_communities, num_communities)) * intra_edge_prob[label] 
#         np.fill_diagonal(community_edge_probability, inter_edge_prob[label])
                
#         for _ in range(num_samples_per_class[label]):

#             node_community = {community: list(range(community * size, (community + 1) * size)) for community, size in enumerate(community_sizes)}
            
#             node_community = node_community_to_onehot(node_community)
#             graph = sample_adjacency(node_community, community_edge_probability)

#             graph_timeseries = {"g0": graph}
#             node_community_timeseries = {"c0": node_community}

#             for time, nodes in enumerate(shuffle_split(node_community[migrate_from_community[label]], time_len-1)):
#                 nodes_from = node_community[migrate_from_community[label]]
#                 nodes_to = node_community[migrate_to_community[label]]

#                 node_community[migrate_from_community[label]] = list(set(nodes_from) - set(nodes))
#                 node_community[migrate_to_community[label]] = nodes_to + nodes

#                 node_community = node_community_to_onehot(node_community)
#                 graph = sample_adjacency(node_community, community_edge_probability)
                
#                 node_community_timeseries["c" + str(time + 1)] = node_community
#                 graph_timeseries ["g" + str(time + 1)] = graph

#             np.savez_compressed(data_dir / ("subject-{:04}_class-{:02}".format(counter, label)), 
#                                 **graph_timeseries, 
#                                 **node_community_timeseries,
#                                 label=label)
        
#             counter += 1


# if __name__ == "__main__":
    # simulate_community_migration_dataset()