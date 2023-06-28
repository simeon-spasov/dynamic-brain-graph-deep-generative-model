import itertools
import math
import random

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


def _gumbel_softmax(logits, device, tau=1, hard=False, eps=1e-10, dim=-1):
    shape = logits.size()
    U = torch.rand(shape).to(device)  # sample from uniform [0, 1)
    g = -torch.log(-torch.log(U + eps) + eps)
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def _bce_loss(p_c_given_z, c, reduction='sum'):
    recon_c_softmax = F.log_softmax(p_c_given_z, dim=-1)
    bce = F.nll_loss(recon_c_softmax, c, reduction=reduction)
    return bce


def _kld_z_loss(q, p_prior):
    log_q = torch.log(q + 1e-20)
    kld = (
        torch.sum(q *
                  (log_q - torch.log(p_prior + 1e-20)), dim=-1)).sum()
    return kld


def _reparameterized_sample(mean, std):
    eps = torch.FloatTensor(std.size()).normal_().to(mean.device)
    return eps.mul(std).add_(mean)


def _kld_gauss(mu_1, std_1, mu_2, std_2_scale):
    std_2 = torch.ones_like(std_1).mul(std_2_scale).to(mu_1.device)
    KLD = 0.5 * torch.sum(
        (2 * torch.log(std_2 + 1e-20) - 2 * torch.log(std_1 + 1e-20) +
         (std_1.pow(2) + (mu_1 - mu_2).pow(2)) / std_2.pow(2) - 1))
    return KLD


def _get_status(index, train_time, valid_time):
    if index < train_time:
        return 'train'
    elif train_time <= index < train_time + valid_time:
        return 'valid'
    else:
        return 'test'


def _divide_graph_snapshots(time_len, valid_prop, test_prop):
    valid_time = math.floor(time_len * valid_prop)
    test_time = math.floor(time_len * test_prop)
    train_time = time_len - valid_time - test_time
    return train_time, valid_time, test_time