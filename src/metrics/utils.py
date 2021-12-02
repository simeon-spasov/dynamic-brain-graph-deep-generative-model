import torch


def reduce(to_reduce, reduction="mean"):
    """aggregates a tensor accross all dimensions by a given method."""

    if reduction == "mean":

        return torch.mean(to_reduce)

    elif reduction == "sum":

        return torch.sum(to_reduce)

    elif reduction == "none":
        
        return to_reduce

    else:

        raise ValueError("reduction '{}' unknown. Choose one of 'mean', 'sum', or 'none'".format(reduction))