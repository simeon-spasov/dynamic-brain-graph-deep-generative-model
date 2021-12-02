import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from src.models.layers import get_activation, GraphConvBlock


MIN_STD, MAX_STD = 1e-4, 10


class Encoder(nn.Module):
    """Encoder for computing node-wise latent variables."""

    def __init__(self, input_dim, hidden_dim, latent_dim, activation=None, batch_norm=True, dropout=0.):

        super(Encoder, self).__init__()

        hidden_dim = [input_dim] + hidden_dim if type(hidden_dim) is list else [input_dim, hidden_dim]
  
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(hidden_dim, hidden_dim[1:])):
            layers += [GraphConvBlock(in_dim, out_dim, activation, batch_norm, dropout)] 
     
        self.gconv_layers = nn.ModuleList(layers)
        self.gconv1 = GraphConvBlock(hidden_dim[-1], latent_dim)
        self.gconv2 = GraphConvBlock(hidden_dim[-1], latent_dim)


    def forward(self, X, A):

        H = X
        for gconv in self.gconv_layers:
            H = gconv(H, A)

        mean = self.gconv1(H, A)
        log_std = self.gconv2(H, A)

        return mean, log_std.exp().clamp(min=MIN_STD, max=MAX_STD)


class InnerProductDecoder(nn.Module):
    """Inner product decoder for computing edge probabilities."""

    def __init__(self, dropout=0.):

        super(InnerProductDecoder, self).__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None 
      

    def forward(self, Z):

        if self.dropout is not None:
            Z = self.dropout(Z)

        A = Z.bmm(Z.transpose(-2, -1))

        return A


class VGAE(nn.Module):
    """..."""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, activation=None, batch_norm=True, dropout=0.):
        
        super(VGAE, self).__init__()

        self.encode = Encoder(input_dim, hidden_dim, latent_dim, activation, batch_norm, dropout)
        self.decode = InnerProductDecoder(dropout)
        self.prior_Z = dist.Normal(loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim))

    
    def forward(self, X, A):

        mean_Z, std_Z = self.encode(X, A)
        posterior_Z = dist.Normal(loc=mean_Z, scale=std_Z)
        Z = posterior_Z.rsample()
        A_recon = self.decode(Z)

        return A_recon, posterior_Z

    
    def loss(self, A_recon, A, posterior_Z):
        
        num_nodes = A.shape[-1]
        num_edges = A.sum(dim=(-2, -1))
        
        # num non edges / num edges
        pos_weight = ((num_nodes ** 2  - num_edges) / num_edges).unsqueeze(-1).unsqueeze(-1)
        # num possible edges / num non edges
        norm = (num_nodes ** 2 / (num_nodes ** 2 - num_edges)).unsqueeze(-1).unsqueeze(-1)

        # BT x N x N
        nll = norm * F.binary_cross_entropy_with_logits(A_recon, A, pos_weight=pos_weight, reduction="none")
        nll = nll.mean()

        # BT x N x K
        kl = dist.kl_divergence(posterior_Z, self.prior_Z.expand(posterior_Z.batch_shape))
        # BT x N x K -> BT x N 
        kl = (1 / num_nodes) * kl.sum(dim=-1)
        kl = kl.mean() 

        elbo = nll + kl
    
        return elbo, nll, kl



