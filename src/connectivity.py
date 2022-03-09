import numpy as np
import torch

from src.utils import is_not_nan_inf


def _to_symmetric(matrix, dim1=-2, dim2=-1):
    """Make a batch of matrices symmetric about diagonal"""
    tril = matrix.tril(diagonal=-1)
    diag = matrix.diagonal(dim1=dim1, dim2=dim2).diag_embed()
    return tril + tril.transpose(dim1, dim2) + diag

def _to_positive_definite(matrix, dim1=-2, dim2=-1, jitter=1e-4, max_iter=100):
    """Add jitter to the diagonal elements of a batch of matrices to ensure positive definiteness"""
    _, info = torch.linalg.cholesky_ex(matrix)
    total = torch.zeros_like(info).float()
    if not torch.any(info):
        return matrix
    else:  
        for i in range(max_iter):
            diag_add = ((info > 0) * jitter).unsqueeze(-1)
            matrix.diagonal(dim1=dim1, dim2=dim2).add_(diag_add)
            total += jitter
            _, info = torch.linalg.cholesky_ex(matrix)
            if not torch.any(info):
                break
        if i + 1 < max_iter:
            return matrix
        else:
            raise ValueError("matrix is not positive definite after adding a max of {} "
                             "to diagonal elements over {} iterations.".format(total, max_iter))

def _remove_self_loops(A, dim1=-2, dim2=-1):
    """Remove self-loops from adjacency matrix"""
    return A - A.diagonal(dim1=dim1, dim2=dim2).diag_embed(offset=0, dim1=dim1, dim2=dim2) 

def _covariance_matrix(X):
    """Compute covariance matrix"""
    mean = X.mean(dim=-1, keepdim=True)
    X_mean = X.sub(mean)
    cov = X_mean.matmul(X_mean.transpose(-2, -1))
    cov = cov / (X.shape[-1] - 1)
    return _to_symmetric(cov)

def correlation_matrix(X):
    """Compute correlation matrix"""
    # covariance matrix
    cov = _covariance_matrix(X)
    # normalize covariance matrix
    std = cov.diagonal(dim1=-2, dim2=-1).pow(0.5)
    corr = cov.div(std.unsqueeze(-1).expand_as(cov))
    corr = corr.div(std.unsqueeze(-1).expand_as(cov).transpose(-2, -1))
    corr = _to_symmetric(corr)
    corr.diagonal(dim1=-2, dim2=-1).fill_(1.)
    return corr.clamp(-1., 1.)

def partial_correlation_matrix(X, epsilon=1e-4):
    """Compute partial correlation matrix"""
    # covariance matrix
    cov = _covariance_matrix(X)
    # jitter to positive definite
    cov = _to_positive_definite(cov, jitter=epsilon)
    # lower cholesky factor
    l = torch.linalg.cholesky(cov)
    # inverse covariance matrix
    cov_inv = torch.cholesky_inverse(l, upper=False)
    # diagonal matrix
    d_inv_sqrt = cov_inv.diagonal(dim1=-2, dim2=-1).pow(-0.5)
    d_inv_sqrt[d_inv_sqrt.isinf()] = 0.
    diag = d_inv_sqrt.diag_embed()
    pcorr = -1 * diag.matmul(cov_inv).matmul(diag)
    pcorr = _to_symmetric(pcorr)
    pcorr.diagonal(dim1=-2, dim2=-1).fill_(1.)
    return pcorr.clip(-1., 1.)

class ConnectivityMeasure:
    """Class for computing different kinds of functional connectivity"""

    fc = {"correlation": correlation_matrix, "partial_correlation": partial_correlation_matrix}
    
    def __init__(self, measure="correlation", self_loops=False):
        self.measure = measure
        self.self_loops = self_loops
        if self.measure not in self.fc.keys():
            raise NotImplementedError("unknown connectivity measure '{}'. " 
                                      "Only {} are supported".format(self.measure, ", ".join(list(self.fc.keys()))))
    
    def __call__(self, X):
        # check dimension of input
        if X.ndim != 3:
            raise ValueError("input must have shape (batch_size, num_nodes, time_len)")
        # compute functional connectivity
        A = self.fc[self.measure](X)
        # error check
        A = is_not_nan_inf(A)
        # remove self loops
        if not self.self_loops:
            A = _remove_self_loops(A)
        return A
