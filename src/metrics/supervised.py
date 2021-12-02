import torch
import torchmetrics.functional as tm
from src.metrics.utils import reduce
# https://github.com/shchur/overlapping-community-detection/blob/master/nocd/metrics/supervised.py


def evaluate_supervised(C_pred, C_true, average="macro"):

    return {"f1": f1(C_pred, C_true, average),
            "precision": precision(C_pred, C_true, average),
            "recall": recall(C_pred, C_true, average),
            "hamming": hamming_distance(C_pred, C_true),
            "jaccard": symmetric_jaccard(C_pred, C_true),
            "nmi": overlapping_nmi(C_pred, C_true)}


def f1(C_pred, C_true, average="macro"):
    """Compute F1 score"""

    # check one_hot

    num_classes = C_true.shape[-1]

    f1_ = tm.f1(C_pred, C_true, num_classes=num_classes, average=average, mdmc_average="global")

    return f1_.item()


def precision(C_pred, C_true, average="macro"):
    """Compute precision"""

    # check one_hot

    num_classes = C_true.shape[-1]

    precision_ = tm.precision(C_pred, C_true, num_classes=num_classes, average=average, mdmc_average="global")

    return precision_.item()


def recall(C_pred, C_true, average="macro"):
    """Compute recall"""

    # check one_hot

    num_classes = C_true.shape[-1]

    recall_ = tm.recall(C_pred, C_true, num_classes=num_classes, average=average, mdmc_average="global")

    return recall_.item()


def hamming_distance(C_pred, C_true):
    """Compute Hamming distance"""

    # check one_hot
    
    hamming = tm.hamming_distance(C_pred, C_true)

    return hamming.item()


def symmetric_jaccard(C_pred, C_true):
    """Compute Jaccard similarity"""

    if C_pred.shape[-1] > C_pred.shape[-2] or C_true.shape[-1] > C_true.shape[-2]:
        raise ValueError("community assigment matrix C must have shape number of nodes (N) by number of communities (K)")

    ## check for ONE HOT !
    ##

    # intersection
    intersection = C_pred.transpose(dim0=-2, dim1=-1).matmul(C_true)
    # union
    union = (C_true.sum(dim=-2).unsqueeze(-2) + C_pred.sum(dim=-2).unsqueeze(dim=-1)) - intersection
    
    # intersection over union
    iou = intersection / union
    iou[union == 0] = 0

    # symmetric jaccard
    jaccard = 0.5 * (iou.max(dim=-2)[0].mean(dim=-1) + iou.max(dim=-1)[0].mean(dim=-1))
    
    return jaccard.mean().clamp(min=0., max=1.).item()


def overlapping_nmi(C_pred, C_true):
    """Compute normalised mutual information"""

    if C_pred.shape[-1] > C_pred.shape[-2] or C_true.shape[-1] > C_true.shape[-2]:
        raise ValueError("community assigment matrix C must have shape number of nodes (N) by number of communities (K)")

    ## check for ONE HOT !
    ##


    def compute_logical_relations(C_pred, C_true):
        """Compute logical relations between sets"""

        # not C_pred and not C_true
        A = ((1 - C_pred) * (1 - C_true)).sum(-1)
        # not C_pred and C_true
        B = ((1 - C_pred) * C_true).sum(-1)
        # C_pred and not C_true
        C = ((1 - C_true) * C_pred).sum(-1)
        # C_pred and C_true 
        D = (C_pred * C_true).sum(-1)
    
        return A, B, C, D


    def h_scalar(c, n):
        """Compute contribution of a single value to the entropy."""

        h = - c * (c / n).log2()
        h[torch.logical_or(h.isinf(), h.isnan())] = 0

        return h
     

    def H_vector(C_pred, C_true):
        """Compute conditional entropy between two vectors."""

        N = C_pred.shape[-1]

        A, B, C, D = compute_logical_relations(C_pred, C_true)
    
        h_a = h_scalar(A, N)
        h_b = h_scalar(B, N)
        h_c = h_scalar(C, N)
        h_d = h_scalar(D, N)

        h_b_d = h_scalar(B + D, N)
        h_a_c = h_scalar(A + C, N)
        h_c_d = h_scalar(C + D, N)
        h_a_b = h_scalar(A + B, N)

        h = torch.zeros_like(h_a)

        idx = h_a + h_d >= h_b + h_c
        if (idx).any():
            h[idx] = (h_a + h_b + h_c + h_d - h_b_d - h_a_c)[idx]

        idx = ~ idx
        if (idx).any():  
            h[idx] = (h_c_d + h_a_b)[idx]

        return h


    def H_uncond(C):
        """compute unconditional entropy."""

        dim = list(C.shape)
        K, N = dim[-2:]

        h = 0
    
        for k in range(K):
        
            h += h_scalar(C[..., k, :].sum(dim=-1), N) 
            h += h_scalar(N - C[..., k, :].sum(dim=-1), N)

        return h


    def H_cond(C_pred, C_true):
        """Compute conditional entropy between two matrices."""

        dim = list(C_pred.shape)
        K = dim[-2]

        h = torch.zeros(*dim[:-2], K, K)

        for k1 in range(K):

            for k2 in range(K):

                h[..., k1, k2] = H_vector(C_pred[..., k1, :], C_true[..., k2, :])

        return h.min(dim=-1)[0].sum(dim=-1)


    # transpose node dim to column
    C_pred = C_pred.transpose(-2, -1)
    C_true = C_true.transpose(-2, -1)
    
    # entropy
    H_C_pred = H_uncond(C_pred)
    H_C_true = H_uncond(C_true)

    # conditional entropy
    H_C_pred_C_true = H_cond(C_pred, C_true)
    H_C_true_C_pred = H_cond(C_true, C_pred)

    # normalizing constant
    norm = torch.cat([H_C_pred.unsqueeze(-1), H_C_true.unsqueeze(-1)], dim=-1)
    
    # mutual information
    I_C_pred_C_true = 0.5 * (H_C_pred + H_C_true)
    I_C_pred_C_true -= 0.5 * (H_C_pred_C_true + H_C_true_C_pred)
    I_C_pred_C_true /= norm.max(dim=-1)[0]
 
    return I_C_pred_C_true.mean().clamp(min=0., max=1.).item()