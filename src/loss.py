import numpy as np
import torch
import torch.nn as nn

# https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py

def get_loss(name, **kwargs):

    if name == "elbo":

        return ELBOLoss(**kwargs)
    
    else:
        raise ValueError("loss not implemented: '{}'".format(name))


class ELBOLoss(nn.Module):
    def __init__(self, gamma_alpha=1., gamma_beta=1., gamma_phi=1., gamma_z=1., edge_balance=True):
        super(ELBOLoss, self).__init__()
     
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_phi = gamma_phi
        self.gamma_z = gamma_z

        self.neg_scale = 1. if edge_balance else num_nonedges / num_edges
            

    def forward(self, nll_pos, nll_neg, kl_alpha, kl_beta, kl_phi, kl_z, warmup=False):
        # sum over time: B x E x T -> B x E 
        nll_pos = nll_pos.sum(dim=-1)
        # mean over number of positive edges and batch
        nll_pol = nll_pos.mean(dim=(-1, -2))

        # sum over time: B x E x T -> B x E 
        nll_neg = nll_neg.sum(dim=-1)
        # mean over number of negative edges and batch
        nll_neg = nllnll_neg_pos.mean(dim=(-1, -2))
        
        # sum over latent dimension: B x Z  -> B
        kl_alpha = kl_alpha.sum(dim=-1)
        # mean over batch
        kl_alpha = kl_alpha.mean(dim=-1)
        
        # sum over latent dimension and time: B x K x Z x T -> B x K 
        kl_beta = kl_beta.sum(dim=(-2, -1))
        # mean over number of communities and batch
        kl_beta = kl_beta.mean(dim=(-1, -2))
        
        # sum over latent dimension and time: B x N x Z x T -> B x N 
        kl_phi = kl_phi.sum(dim=(-2, -1))
        # mean over number of nodes and batch
        kl_phi = kl_phi.mean(dim=(-1, -2))

        # sum over latent dimension and time: B x E x K x T -> B x E 
        kl_z = kl_z.sum(dim=(-2, -1))
        # mean over number of edges and batch
        kl_z = kl_z.mean(dim=(-2, -1))

        loss = (nll_pos + (self.neg_scale * nll_neg)) / (1 + self.neg_scale)

        if not warmup:
            loss += self.gamma_alpha * kl_alpha
            loss += self.gamma_beta * kl_beta
            loss += self.gamma_phi * kl_phi
            loss += self.gamma_z * kl_z
            
            loss_parts = np.array((nll_pos.item(), nll_neg.item(), kl_alpha.item(), kl_beta.item(), kl_phi.item(), kl_z.item()))
        
        else:
            
            loss_parts = np.array([0.] * 6)
                
        return loss, loss_parts