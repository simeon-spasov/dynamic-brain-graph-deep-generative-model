import random
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


def set_random_seed(seed, is_gpu=False):
    """Set random seeds for reproducability"""
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if not (min_seed_value <= seed <= max_seed_value):
        raise ValueError("seed '{}' is not in [{} to {}]".format(seed, min_seed_value, max_seed_value))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and is_gpu:
        torch.cuda.manual_seed_all(seed)

def set_device(is_gpu=True, gpu_number=0):
    """Set device"""
    gpu_count = torch.cuda.device_count()
    if gpu_count < gpu_number:
        raise ValueError("number of cuda devices {}".format(gpu_count))
    else:
        if torch.cuda.is_available() and is_gpu:
            device = torch.device("cuda:{}".format(gpu_number))
        else:
            device = torch.device("cpu")
    return device

def load_config(filename, config_dir="./config"):
    """Load config file"""
    filepath = Path(config_dir) / (filename + ".yaml")
    try:
        config = OmegaConf.load(filepath)
        return config
    except:
        raise IOError("error loading config '{}'".format(str(filepath)))

def save_config(config, save_dir):
    """Save config file"""
    filepath = Path(save_dir) / "config.yaml"
    try:
        if isinstance(config, dict):
            with open(save_dir / filename, "w") as outfile:
                yaml.dump(config, outfile)
        else:
            OmegaConf.save(config, filepath)
    except:
        raise IOError("error saving config to '{}'".format(str(save_dir)))

def is_not_nan_inf(tensor):
    """Check elements of tensor for inf or nan"""
    if (~tensor.isnan()).all() and (~tensor.isinf()).all():
        return tensor
    else: 
        raise ValueError("tensor elements are inf or nan")

def adjacency_to_edge_weights(A, tril=True):
    """Reperesnt adjacency matrix as edge indices and weights"""
    batch_size = A.shape[0]
    # zero out upper triangle including diagonal
    if tril:
        A = A.tril(diagonal=-1)
    # (batch_size, num_nodes, num_nodes) -> (batch_size, num_edges, 2)
    edges = A.nonzero().reshape(batch_size, -1, 3)[..., 1:]
    # (batch_size, num_nodes, num_nodes) -> (batch_size, num_edges)
    weights = A[A.nonzero(as_tuple=True)].reshape(batch_size, -1)
    # sanity check
    assert edges.shape[:-1] == weights.shape
    return edges, weights
