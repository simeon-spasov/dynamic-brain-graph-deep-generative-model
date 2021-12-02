import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from src.models.layers import get_activation, LinearBlock, GraphConvBlock, GraphConvLSTMCell
from src.models.VGAE import InnerProductDecoder


MIN_STD, MAX_STD = 1e-5, 10


class GraphConvLSTM(nn.Module):
    """..."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, bias=True):
        
        super(GraphConvLSTM, self).__init__()

        hidden_dim = [input_dim] + hidden_dim * num_layers if isinstance(hidden_dim, list) else [input_dim] + [hidden_dim] * num_layers

        layers = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(zip(hidden_dim, hidden_dim[1:])):
            layers += [GraphConvLSTMCell(in_dim, out_dim, bias)]

        self.gconv_layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim[-1]
        self.num_layers = num_layers
    

    def init_hidden_state(self, X, hidden_state=None):

        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, X.shape[0], X.shape[-2], self.hidden_dim, dtype=X.dtype, device=X.device)
        
        return hidden_state


    def init_cell_state(self, X, cell_state=None):

        if cell_state is None:
            cell_state = torch.zeros(self.num_layers, X.shape[0], X.shape[-2], self.hidden_dim, dtype=X.dtype, device=X.device)
        
        return cell_state

    
    def forward(self, X, A, hidden_state, cell_state):
        
        out = X

        _hidden_state = []
        _cell_state = []

        for idx, gconv in enumerate(self.gconv_layers):
            
            H, C = gconv(out, A, hidden_state[idx], cell_state[idx])
            _hidden_state += [H]
            _cell_state += [C]

            out = H

        return torch.stack(_hidden_state), torch.stack(_cell_state)


class RVGAE(nn.Module):
    """..."""

    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, num_layers=1, batch_norm=True, dropout=0.):
        
        super(RVGAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self._create_layers()


    def _create_prior_Z_params(self):

        self.fc_p_Z = LinearBlock(self.hidden_dim, self.hidden_dim, activation="relu", batch_norm=self.batch_norm)
        self.fc_p_Z_mean = LinearBlock(self.hidden_dim, self.latent_dim, activation=None)
        self.fc_p_Z_std = LinearBlock(self.hidden_dim, self.latent_dim, activation="softplus")


    def _create_posterior_Z_params(self):

        self.fc_phi_X = LinearBlock(self.input_dim, self.hidden_dim, activation="relu", batch_norm=self.batch_norm)    
        self.gconv_q_Z = GraphConvBlock(self.hidden_dim * 2, self.hidden_dim, activation=None, batch_norm=self.batch_norm)
        self.gconv_q_Z_mean = GraphConvBlock(self.hidden_dim, self.latent_dim, activation=None)
        self.gconv_q_Z_std = GraphConvBlock(self.hidden_dim, self.latent_dim, activation="softplus")

    
    def _create_A_likelihood_params(self):

        self.fc_phi_Z = LinearBlock(self.latent_dim, self.hidden_dim, activation="relu", batch_norm=self.batch_norm)
        self.decoder = InnerProductDecoder()


    def _create_layers(self):
        
        self._create_prior_Z_params()
        self._create_posterior_Z_params()
        self._create_A_likelihood_params()
        self.rnn = GraphConvLSTM(self.hidden_dim * 2, self.hidden_dim, self.num_layers, bias=True)
    

    def _parameterise_prior_Z(self, hidden_state):

        H = self.fc_p_Z(hidden_state)
        p_Z_mean = self.fc_p_Z_mean(H)
        p_Z_std = self.fc_p_Z_std(H).clamp(min=MIN_STD, max=MAX_STD)
        p_Z = dist.Normal(loc=p_Z_mean, scale=p_Z_std)

        return p_Z


    def _parameterise_posterior_Z(self, X, A, hidden_state):

        phi = self.fc_phi_X(X)
        H = torch.cat([phi, hidden_state], dim=2)
        H = self.gconv_q_Z(H, A)
        q_Z_mean = self.gconv_q_Z_mean(H, A)
        q_Z_std = self.gconv_q_Z_std(H, A).clamp(min=MIN_STD, max=MAX_STD)
        q_Z = dist.Normal(loc=q_Z_mean, scale=q_Z_std)

        return q_Z, phi

    
    def _parameterise_likelihood_A(self, Z):

        phi = self.fc_phi_Z(Z)
        logits = self.decoder(Z)
        p_A = dist.Bernoulli(logits=logits)

        return p_A, phi

    
    def forward(self, X, A):

        time_dim = X.shape[-3]

        p_Z_1_T = []
        q_Z_X_1_T = []
        p_A_Z_1_T_sample = []

        H_t = self.rnn.init_hidden_state(X)
        C_t = self.rnn.init_cell_state(X)

        for t in range(time_dim):

            X_t = X[:, t, ...]
            A_t = A[:, t, ...]
  
            # parameterise prior Z
            p_Z_t = self._parameterise_prior_Z(H_t[-1])
            # parameterise posterior Z given X
            q_Z_X_t, phi_X_t = self._parameterise_posterior_Z(X_t, A_t, H_t[-1])
            # sample posterior Z given X
            q_Z_X_t_sample = q_Z_X_t.rsample()
            # parameterise likelihood A given Z
            p_A_Z_t, phi_Z_t = self._parameterise_likelihood_A(q_Z_X_t_sample)
            # sample likelihood A given Z
            p_A_Z_t_sample = p_A_Z_t.mean

            # accumulate prior Z, posterior Z given X, and likelihood A given Z
            p_Z_1_T += [p_Z_t]
            q_Z_X_1_T += [q_Z_X_t]
            p_A_Z_1_T_sample += [p_A_Z_t_sample]

            # recurrence
            phi_t = torch.cat([phi_X_t, phi_Z_t], dim=2)
            H_t, C_t = self.rnn(phi_t, A_t, H_t, C_t)
        
        return torch.stack(p_A_Z_1_T_sample, dim=1), q_Z_X_1_T, p_Z_1_T
            
    
    def loss(self, A_recon, A, posterior_Z, prior_Z):
        
        num_nodes = A.shape[-1]
        num_edges = A.sum(dim=(-2, -1))
        
        # num non edges / num edges
        pos_weight = ((num_nodes ** 2  - num_edges) / num_edges).unsqueeze(-1).unsqueeze(-1)
        # num possible edges / num non edges
        norm = (num_nodes ** 2 / (num_nodes ** 2 - num_edges)).unsqueeze(-1).unsqueeze(-1)

        # B X T x N x N
        nll = norm * F.binary_cross_entropy_with_logits(A_recon, A, pos_weight=pos_weight, reduction="none")
        nll = nll.mean()

        # B x T x N x K
        kl = torch.stack([dist.kl_divergence(q_Z, p_Z) for q_Z, p_Z in zip(posterior_Z, prior_Z)], dim=1)
        # B x T x N x K -> B x T x N 
        kl = (1 / num_nodes) * kl.sum(dim=-1)
        kl = kl.mean() 

        elbo = nll + kl
    
        return elbo, nll.item(), kl.item()



        
        

