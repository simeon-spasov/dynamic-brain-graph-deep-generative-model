import itertools
import math
import random

import torch
import torch.nn.functional as F

import random
import itertools
import networkx as nx
from typing import List, Tuple


def sample_pos_neg_edges(graph: nx.Graph, num_samples: int = 1) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Samples positive and negative edges from a networkx graph for the evaluation of a graph representation learning algorithm.

    Args:
        graph (nx.Graph): The graph from which to sample edges.
        num_samples (int, optional): The number of samples to take. Default is 1.

    Returns:
        Tuple[List[Tuple], List[Tuple]]: A tuple of two lists. The first list contains tuples representing the positive
        edges sampled from the graph. The second list contains tuples representing the negative edges sampled.

    Raises:
        ValueError: If `num_samples` is greater than the number of available negative edges.
    """

    # Extract the set of positive edges from the graph
    pos_edges = set(graph.edges())

    # Generate the set of all possible edges (both positive and negative)
    all_edges = set(itertools.combinations(graph.nodes(), 2))

    # Calculate the set of negative edges by subtracting the positive edges from all possible edges
    all_neg_edges = all_edges - pos_edges

    if num_samples > len(all_neg_edges):
        raise ValueError("The number of requested samples exceeds the number of available negative edges.")

    # Sample the same number of negative edges as there are positive edges, `num_samples` times
    sampled_neg_edges = [random.sample(all_neg_edges, len(pos_edges)) for _ in range(num_samples)]

    # Repeat the list of positive edges `num_samples` times
    sampled_pos_edges = [list(graph.edges()) for _ in range(num_samples)]

    # Flatten the lists of sampled edges
    return list(itertools.chain.from_iterable(sampled_pos_edges)), \
           list(itertools.chain.from_iterable(sampled_neg_edges))


def gumbel_softmax(logits: torch.Tensor, device: torch.device, tau: float = 1.0, hard: bool = False,
                   eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    """
    Applies the Gumbel-Softmax trick for sampling from categorical distributions.

    Args:
        logits (Tensor): Input tensor representing categorical distributions.
        device (torch.device): The device on which tensors should be allocated.
        tau (float, optional): Temperature parameter for Gumbel-Softmax. Default is 1.0.
        hard (bool, optional): If True, the returned samples will be one-hot encoded. Default is False.
        eps (float, optional): Small value to prevent numerical instability. Default is 1e-10.
        dim (int, optional): The dimension along which softmax will be computed. Default is -1 (last dimension).

    Returns:
        Tensor: Tensor of same shape as `logits` representing the Gumbel-Softmax sampled tensors.
    """
    shape = logits.size()

    # Uniform random numbers, shifted to device
    U = torch.rand(shape).to(device)

    # Gumbel distributed noise
    g = -torch.log(-torch.log(U + eps) + eps)

    # Softmax with Gumbel noise and temperature
    y_soft = F.softmax((logits + g) / tau, dim=dim)

    if hard:
        # Straight through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick
        ret = y_soft

    return ret


def bce_loss(p_c_given_z: torch.Tensor, c: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
    """
    Computes the Binary Cross-Entropy (BCE) loss.

    Args:
        p_c_given_z (Tensor): Predicted probabilities.
        c (Tensor): Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output. Default is 'sum'.

    Returns:
        Tensor: The BCE loss.
    """
    recon_c_softmax = F.log_softmax(p_c_given_z, dim=-1)
    bce = F.nll_loss(recon_c_softmax, c, reduction=reduction)
    return bce


def kld_z_loss(q: torch.Tensor, p_prior: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler Divergence (KLD) loss.

    Args:
        q (Tensor): The approximate posterior probability distribution.
        p_prior (Tensor): The prior probability distribution.

    Returns:
        Tensor: The KLD loss.
    """
    log_q = torch.log(q + 1e-20)
    kld = (torch.sum(q * (log_q - torch.log(p_prior + 1e-20)), dim=-1)).sum()
    return kld


def reparameterized_sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Samples from a Gaussian distribution using the reparameterization trick.

    Args:
        mean (Tensor): The mean values of the Gaussian distribution.
        std (Tensor): The standard deviations of the Gaussian distribution.

    Returns:
        Tensor: A tensor sampled from the Gaussian distribution defined by the mean and standard deviation.
    """
    eps = torch.randn_like(std).to(mean.device)
    return eps.mul(std).add_(mean)


def kld_gauss(mu_1: torch.Tensor, std_1: torch.Tensor, mu_2: torch.Tensor, std_2_scale: float) -> torch.Tensor:
    """
    Computes the Kullback-Leibler Divergence (KLD) between two Gaussian distributions.

    Args:
        mu_1 (Tensor): Mean of the first Gaussian distribution.
        std_1 (Tensor): Standard deviation of the first Gaussian distribution.
        mu_2 (Tensor): Mean of the second Gaussian distribution.
        std_2_scale (float): Standard deviation scale of the second Gaussian distribution.

    Returns:
        Tensor: The KLD between the two Gaussian distributions.
    """
    std_2 = torch.ones_like(std_1).mul(std_2_scale).to(mu_1.device)
    KLD = 0.5 * torch.sum(
        (2 * torch.log(std_2 + 1e-20) - 2 * torch.log(std_1 + 1e-20) +
         (std_1.pow(2) + (mu_1 - mu_2).pow(2)) / std_2.pow(2) - 1))
    return KLD


def get_status(index: int, train_time: int, valid_time: int) -> str:
    """
    Determines the status of a data point based on its index.

    Args:
        index (int): The index of the data point.
        train_time (int): The number of data points allocated for training.
        valid_time (int): The number of data points allocated for validation.

    Returns:
        str: The status of the data point - 'train', 'valid', or 'test'.
    """
    if index < train_time:
        return 'train'
    elif train_time <= index < train_time + valid_time:
        return 'valid'
    else:
        return 'test'


def divide_graph_snapshots(time_len, valid_prop, test_prop):
    """
    Divides graph snapshots into training, validation, and testing sets based on the provided proportions.

    Args:
        time_len (int): Total number of graph snapshots.
        valid_prop (float): Proportion of the graph snapshots to use for validation.
        test_prop (float): Proportion of the graph snapshots to use for testing.

    Returns:
        tuple: The number of graph snapshots for training, validation, and testing sets.

    Raises:
        ValueError: If `valid_prop` and `test_prop` don't add up to less than 1.
        ValueError: If `valid_time`, `test_time`, or `train_time` is not greater than 0.
    """

    if valid_prop + test_prop >= 1:
        raise ValueError("Sum of `valid_prop` and `test_prop` must be less than 1.")

    valid_time = math.floor(time_len * valid_prop)
    test_time = math.floor(time_len * test_prop)
    train_time = time_len - valid_time - test_time

    if valid_time <= 0 or test_time <= 0 or train_time <= 0:
        raise ValueError("The calculated `train_time`, `valid_time`, or `test_time` is not greater than 0. "
                         "You might have to adjust your window size or stride.")

    return train_time, valid_time, test_time

