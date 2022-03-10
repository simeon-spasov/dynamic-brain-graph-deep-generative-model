import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import (Categorical, Normal, OneHotCategorical,
                                 kl_divergence)

from src.connectivity import ConnectivityMeasure
from src.utils import adjacency_to_edge_weights, is_not_nan_inf


class Model(nn.Module):
    def __init__(self, num_samples, num_nodes, num_communities=8, alpha_dim=14,
                 beta_dim=9, phi_dim=23, alpha_std=0.01, temp=1., window_size=10,
                 window_stride=2, measure="correlation", percentile=5., device="cpu"):
        super().__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_communities = num_communities

        self.alpha_dim = alpha_dim
        self.alpha_std = alpha_std
        self.beta_dim = beta_dim
        self.phi_dim = phi_dim
        self.temp = temp

        self.window_size = window_size
        self.window_stride = window_stride
        self.connectivity_measure = ConnectivityMeasure(measure)
        self.percentile = percentile

        self.device = device

        self.q_beta = []
        self.q_phi = []
        self.inference = False
        
        self._initialize_params()
        self.to(device)

    def _create_alpha_prior_params(self):
        self.p_alpha_mean = torch.zeros(self.alpha_dim).to(self.device)
        self.p_alpha_std = (torch.ones(self.alpha_dim) * self.alpha_std).to(self.device)

    def _create_alpha_posterior_params(self):    
        self.h_q_alpha = nn.Embedding(self.num_samples, self.alpha_dim, padding_idx=0)
        self.fc_q_alpha = nn.Linear(self.alpha_dim, 2 * self.alpha_dim, bias=True)
            
    def _create_beta_prior_params(self):
        # dense layer to transform alpha to dimension of beta
        # (batch_size, alpha_dim) -> (batch_size, num_communities, beta_dim)
        self.fc_p_alpha_beta = nn.Linear(self.alpha_dim, self.beta_dim * self.num_communities, bias=True)
        # initial hidden state for gru
        self.h_p_beta = torch.zeros(1, self.num_communities, self.beta_dim).to(self.device)
        # gru
        self.gru_p_beta = nn.GRU(self.beta_dim, self.beta_dim, num_layers=1, batch_first=True, bias=True)
        # dense layer to parameterize mean and std from hidden state of gru
        self.fc_p_beta = nn.Linear(self.beta_dim, 2 * self.beta_dim, bias=True)

    def _create_beta_posterior_params(self):
        # dense layer to transform alpha to dimension of beta
        # (batch_size, alpha_dim) -> (batch_size, num_communities, beta_dim)
        self.fc_q_alpha_beta = nn.Linear(self.alpha_dim, self.beta_dim * self.num_communities, bias=True)
        # initial hidden state
        self.h_q_beta = torch.zeros(1, self.num_communities, self.beta_dim).to(self.device)
        # gru
        self.gru_q_beta = nn.GRU(self.beta_dim, self.beta_dim, num_layers=1, batch_first=True, bias=True)
        # dense layer to parameterize mean and std from hidden state of gru
        self.fc_q_beta = nn.Linear(self.beta_dim, 2 * self.beta_dim, bias=True)

    def _create_phi_prior_params(self):
        self.fc_p_alpha_phi = nn.Linear(self.alpha_dim, self.phi_dim * self.num_nodes, bias=True)
        self.h_p_phi = torch.zeros(1, self.num_nodes, self.phi_dim).to(self.device)
        self.gru_p_phi = nn.GRU(self.phi_dim, self.phi_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_p_phi = nn.Linear(self.phi_dim, 2 * self.phi_dim, bias=True)
        
    def _create_phi_posterior_params(self):
        self.fc_q_alpha_phi = nn.Linear(self.alpha_dim, self.phi_dim * self.num_nodes, bias=True)
        self.h_q_phi = torch.zeros(1, self.num_nodes, self.phi_dim).to(self.device)
        self.gru_q_phi = nn.GRU(self.phi_dim, self.phi_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_q_phi = nn.Linear(self.phi_dim, 2 * self.phi_dim, bias=True)

    def _create_z_prior_params(self):
        self.fc_p_z = nn.Linear(self.phi_dim, self.num_communities, bias=True)
        
    def _create_z_posterior_params(self):
        self.fc_q_z = nn.Linear(2 * self.phi_dim, self.num_communities, bias=True)

    def _create_c_likelihood_params(self):
        self.fc_p_c = nn.Linear(self.beta_dim, self.num_nodes, bias=True)
    
    def _initialize_params(self):
        self._create_alpha_prior_params()
        self._create_beta_prior_params()
        self._create_phi_prior_params()
        self._create_z_prior_params()
        self._create_c_likelihood_params()
        self._create_alpha_posterior_params()
        self._create_beta_posterior_params()
        self._create_phi_posterior_params()
        self._create_z_posterior_params()
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -1.0, 1.0)
            if isinstance(module, nn.GRU):
                pass

    # prior over subject embeddings
    # p(alpha) = Normal((mean, std) = (0, 0.01))
    def _parameterize_prior_alpha(self, sample_idx):
        batch_size = sample_idx.shape[0] if sample_idx is not None else self.num_samples
        mean = self.p_alpha_mean
        std = self.p_alpha_std
        return Normal(mean, std).expand(torch.Size([batch_size, self.alpha_dim]))

    # posterior over subject embeddings
    # q(alpha) = Normal((mean, std) = NN(E)))
    def _parameterize_posterior_alpha(self, sample_idx):
        # subject embedding
        # (batch_size, alpha_dim)
        out = self.h_q_alpha(sample_idx).squeeze(1) if sample_idx is not None else self.h_q_alpha.weight
        # (batch_size, alpha_dim) -> (batch_size, 2 * alpha_dim) 
        out = self.fc_q_alpha(out)
        # (batch_size, alpha_dim)
        mean = out[..., :self.alpha_dim]
        # (batch_size, alpha_dim)
        std = F.softplus(out[..., self.alpha_dim:])
        return Normal(mean, std)

    # prior over community embeddings parameterized from subject embeddings and hidden state
    # p(beta_t|alpha, h_1:t-1) = Normal((mean_t, std_t) = GRU(alpha, h_1:t-1))
    def _parameterize_prior_beta(self, alpha, hidden):
        # (batch_size, alpha_dim) -> (batch_size, num_communities * beta_dim)
        input = self.fc_p_alpha_beta(alpha)
        # (batch_size, num_communities * beta_dim) -> (batch_size * num_communities, 1, beta_dim)
        input = input.reshape(-1, 1, self.beta_dim)
        # (batch_size * num_communities, 1, beta_dim), (1, batch_size * num_communities, beta_dim) -> (1, batch_size * num_communities, beta_dim)
        _, hidden = self.gru_p_beta(input, hidden)
        # (1, batch_size * num_communities, beta_dim) -> (batch_size, num_communities, beta_dim)
        hidden = hidden.reshape(-1, self.num_communities, self.beta_dim)
        # (batch_size, num_communities, beta_dim) -> (batch_size, num_communities, 2 * beta_dim)
        out = self.fc_p_beta(hidden)
        # (batch_size, num_communities, beta_dim)
        mean = out[..., :self.beta_dim]
        # (batch_size, num_communities, beta_dim)
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)

    # posterior over community embeddings parameterized from subject embeddings and hidden state
    # q(beta_t|alpha, h_1:t-1) = Normal((mean_t, std_t) = GRU(alpha, h_1:t-1))
    def _parameterize_posterior_beta(self, alpha, hidden):
        # (batch_size, alpha_dim) -> (batch_size, num_communities * beta_dim)
        input = self.fc_q_alpha_beta(alpha)
        # (batch_size, num_communities * beta_dim) -> (batch_size * num_communities, 1, beta_dim)
        input = input.reshape(-1, 1, self.beta_dim)
        # (batch_size * num_communities, 1, beta_dim), (1, batch_size * num_communities, beta_dim) -> (1, batch_size * num_communities, beta_dim)
        _, hidden = self.gru_q_beta(input, hidden)
        # (1, batch_size * num_communities, beta_dim) -> (batch_size, num_communities, beta_dim)
        hidden = hidden.reshape(-1, self.num_communities, self.beta_dim)
        # (batch_size, num_communities, beta_dim) -> (batch_size, num_communities, 2 * beta_dim)
        out = self.fc_q_beta(hidden)
        # (batch_size, num_communities, beta_dim)
        mean = out[..., :self.beta_dim]
        # (batch_size, num_communities, beta_dim)
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)

    # prior over node embeddings parameterized from subject embedding and hidden state
    # p(phi_t|alpha, h_1:t-1) = Normal((mean_t, std_t) = GRU(alpha, h_1:t-1))
    def _parameterize_prior_phi(self, alpha, hidden):
        # (batch_size, alpha_dim) -> (batch_size, num_nodes * phi_dim)
        input = self.fc_p_alpha_phi(alpha)
        # (batch_size, num_nodes * phi_dim) -> (batch_size * num_nodes, 1, phi_dim)
        input = input.reshape(-1, 1, self.phi_dim)
        # (batch_size * num_nodes, 1, phi_dim), (1, batch_size * num_nodes, phi_dim) -> (1, batch_size * num_nodes, phi_dim)
        _, hidden = self.gru_p_phi(input, hidden)
        # (1, batch_size * num_nodes, phi_dim) -> (batch_size, num_nodes, phi_dim)
        hidden = hidden.reshape(-1, self.num_nodes, self.phi_dim)
        # (batch_size, num_nodes, phi_dim) -> (batch_size, num_nodes, 2 * phi_dim)
        out = self.fc_p_phi(hidden)
        # (batch_size, num_nodes, phi_dim)
        mean = out[..., :self.phi_dim]
        # (batch_size, num_nodes, phi_dim)
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)

    # posterior over node embeddings parameterized from subject embedding and hidden state
    # q(phi_t|alpha, h_1:t-1) = Normal((mean_t, std_t) = GRU(alpha, h_1:t-1))
    def _parameterize_posterior_phi(self, alpha, hidden):
        # (batch_size, alpha_dim) -> (batch_size, num_nodes * phi_dim)
        input = self.fc_q_alpha_phi(alpha)
        # (batch_size, num_nodes * phi_dim) -> (batch_size * num_nodes, 1, phi_dim)
        input = input.reshape(-1, 1, self.phi_dim)
        # (batch_size * num_nodes, 1, phi_dim), (1, batch_size * num_nodes, phi_dim) -> (1, batch_size * num_nodes, phi_dim)
        _, hidden = self.gru_q_phi(input, hidden)
        # (1, batch_size * num_nodes, phi_dim) -> (batch_size, num_nodes, phi_dim)
        hidden = hidden.reshape(-1, self.num_nodes, self.phi_dim)
        # (batch_size, num_nodes, phi_dim) -> (batch_size, num_nodes, 2 * phi_dim)
        out = self.fc_q_phi(hidden)
        # (batch_size, num_nodes, phi_dim)
        mean = out[..., :self.phi_dim]
        # (batch_size, num_nodes, phi_dim)
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)

    # prior over community assignment
    # p(z_t|phi_t, w_t) = NN(phi_t, w_t) 
    def _parameterize_prior_z(self, p_phi_sample, w):
        # (batch_size, num_nodes, phi_dim) -> (batch_size, num_edges, phi_dim)
        phi_w = p_phi_sample.gather(dim=1, index=w.unsqueeze(-1).expand((*list(w.shape), self.phi_dim)))
        # (batch_size, num_edges, phi_dim) -> (batch_size, num_edges, num_communities) 
        logits = self.fc_p_z(phi_w)
        return OneHotCategorical(logits=logits)

    # posterior over community assignment
    # q(z_t| phi_t, w_t, c_t) = NN(phi_t, w_t, c_t) 
    def _parameterize_posterior_z(self, q_phi_sample, w, c):
        # embedding source node
        # (batch_size, num_nodes, phi_dim) -> (batch_size, num_edges, phi_dim)
        phi_w = q_phi_sample.gather(dim=1, index=w.unsqueeze(-1).expand((*list(w.shape), self.phi_dim)))
        # embedding target node
        # (batch_size, num_nodes, phi_dim) -> (batch_size, num_edges, phi_dim)
        phi_c = q_phi_sample.gather(dim=1, index=c.unsqueeze(-1).expand((*list(c.shape), self.phi_dim)))
        # (batch_size, num_edges, phi_dim) -> (batch_size, num_edges, 2 * phi_dim)
        h = torch.cat((phi_w, phi_c), dim=-1)
        # (batch_size, num_edges, 2 * phi_dim) -> (batch_size, num_edges, num_communities)
        logits = self.fc_q_z(h)
        return OneHotCategorical(logits=logits)

    # conditional likelihood of edges
    # p(c_t | z_t, beta_t) =
    def _parameterize_likelihood_c(self, beta, z):
        # select community embeddings for each edge
        # (batch_size, num_edges, num_communities), (batch_size, num_communities, beta_dim) -> (batch_size, num_edges, beta_dim)
        beta_z = z.bmm(beta)
        # (batch_size, num_edges, beta_dim) -> (batch_size, num_edges, num_nodes)
        logits = self.fc_p_c(beta_z)
        return Categorical(logits=logits)

    def _dynamic_connectivity_matrix(self, x):
        batch_size, num_timesteps = x.shape[0], x.shape[-1]
        # pad time dimension in order to split into windows 
        if (num_timesteps % self.window_stride == 0):
            pad = max(self.window_size - self.window_stride, 0)
        else:
            pad = max(self.window_size - (num_timesteps % self.window_stride), 0)
        pad_left = pad // 2
        pad_right = pad - pad_left
        x = F.pad(x, (pad_left, pad_right)) 
        # split into windows
        # (batch_size, num_nodes, num_timesteps) -> (batch_size, num_nodes, num_windows, window_size)
        x = x.unfold(-1, self.window_size, self.window_stride)
        num_windows = x.shape[-2]
        # (batch_size, num_nodes, num_windows, window_size) -> (batch_size * num_windows, num_nodes, window_size)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.num_nodes, self.window_size)

        # compute connectivity between nodes in each window
        # (batch_size * num_windows, num_nodes, window_size) -> (batch_size * num_windows, num_nodes, num_nodes)
        A = self.connectivity_measure(x)

        if self.percentile is not None:
            # (batch_size * num_windows, num_nodes, num_nodes) -> (batch_size * num_windows, num_nodes * num_nodes)
            A = A.reshape(-1, self.num_nodes * self.num_nodes) #.type(torch.float64).abs()

            percentile = (100. - self.percentile) / 100.
            # threshold above which p percent of values in the frequency distribution values falls
            threshold = A.quantile(percentile, dim=-1).unsqueeze(-1) 
            # # print(threshold[0], threshold[0].dtype)
            A[A <= threshold] = 0.
        # (batch_size * num_windows, num_nodes, num_nodes) -> (batch_size, num_windows, num_nodes, num_nodes)
        return A.reshape(batch_size, -1, self.num_nodes, self.num_nodes).type(torch.float32)
  
    def _to_edge_weights(self, A):
        batch_size, num_timesteps, num_nodes, _ = A.shape
        # (batch_size, num_timesteps, num_nodes, num_nodes) -> (batch_size * num_timesteps, num_nodes, num_nodes)
        A = A.reshape(-1, num_nodes, num_nodes)
        # (batch_size * num_timesteps, num_edges, 2), (batch_size * num_timesteps, num_edges)
        edges, weights = adjacency_to_edge_weights(A)
        # shuffle
        if self.training:
            idx = torch.randperm(edges.shape[-2]).to(self.device)
            edges, weights = edges[..., idx, :], weights[..., idx]
        # (batch_size * num_timesteps, num_edges, 2) -> (batch_size, num_timesteps, num_edges, 2)
        edges = edges.reshape(batch_size, num_timesteps, -1, 2)
        # (batch_size * num_timesteps, num_edges) -> (batch_size, num_timesteps, num_edges)
        weights = weights.reshape(batch_size, num_timesteps, -1)
        return edges, weights
    
    def _forward_edges(self, edges, weights, p_phi_sample, q_phi_sample, q_beta_sample):
        # unpack edges 
        # (batch_size, num_edges, 2) -> (batch_size, num_edges), (batch_size, num_edges)
        w, c = edges[..., 0], edges[..., 1]
        # parameterize prior p(z | w)
        p_z = self._parameterize_prior_z(p_phi_sample, w)
        # parameterize posterior q(z | w, c)
        q_z = self._parameterize_posterior_z(q_phi_sample, w, c)  
        # kl divergence KL(q(z | w, c)||p(z | w))
        # (batch_size, num_edges)
        kl_z = kl_divergence(q_z, p_z)

        # sample posterior q(z | w, c)
        # (batch_size, num_edges, num_communities) 
        q_z_sample = F.gumbel_softmax(q_z.logits, self.temp, hard=True) if self.training else F.one_hot(q_z.logits.argmax(-1), num_classes=self.num_communities).float()  
        # parameterize likelihood p(c | z, beta)
        p_c = self._parameterize_likelihood_c(q_beta_sample, q_z_sample)
        # negative log likelihood = cross-entropy
        # (batch_size * num_edges, num_nodes), (batch_size * num_edges, ) -> (batch_size * num_edges, )
        nll = F.cross_entropy(p_c.logits.reshape(-1, self.num_nodes), c.reshape(-1), reduction="none")
        # (batch_size * num_edges, ) -> (batch_size, num_edges)
        nll = nll.reshape(q_z_sample.shape[0], -1)
        return nll, kl_z
    
    def _forward_timesteps(self, edges, weights, p_alpha_sample, q_alpha_sample):
        batch_size, num_windows = edges.shape[:2]
        # (1, num_communities, beta_dim) -> (1, batch_size * num_communities, beta_dim)
        # first dim is 1 as bidirectional = False and num_layers = 1 in gru
        h_p_beta_t = self.h_p_beta.repeat_interleave(batch_size, dim=1) 
        h_q_beta_t = self.h_q_beta.repeat_interleave(batch_size, dim=1) 
        # (1, num_nodes, phi_dim) -> (1, batch_size * num_nodes, phi_dim)
        # first dim is 1 as bidirectional = False and num_layers = 1 in gru
        h_p_phi_t = self.h_p_phi.repeat_interleave(batch_size, dim=1) 
        h_q_phi_t = self.h_q_phi.repeat_interleave(batch_size, dim=1) 

        # store elbo loss parts over time
        nll, kl_beta, kl_phi, kl_z = [], [], [], []
        
        # loop over time
        for t in range(num_windows):
            # parameterize prior p(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            p_beta_t = self._parameterize_prior_beta(p_alpha_sample, h_p_beta_t)
            # parameterize posterior q(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            q_beta_t = self._parameterize_posterior_beta(q_alpha_sample, h_q_beta_t)
            # calculate kl divergence KL(q(beta_t|beta_t-1)||p(beta_t|beta_t-1))
            # (batch_size, num_communities, beta_dim)
            kl_beta += [kl_divergence(q_beta_t, p_beta_t)]
           
            # sample prior beta_t ~ p(beta_t|beta_t-1) 
            # (batch_size, num_communities, beta_dim)
            p_beta_sample_t = p_beta_t.mean
            # sample posterior beta_t ~ q(beta_t|beta_t-1) 
            # (batch_size, num_communities, beta_dim)
            q_beta_sample_t = q_beta_t.rsample() if self.training else q_beta_t.mean
    
            # update hidden state beta prior h_t = beta_t
            # (1, num_communities * batch_size, beta_dim)
            h_p_beta_t = p_beta_sample_t.reshape(1, -1, self.beta_dim).contiguous() 
            # update hidden state beta posterior h_t = beta_t
            # (1, num_communities * batch_size, beta_dim)
            h_q_beta_t = q_beta_sample_t.reshape(1, -1, self.beta_dim).contiguous()  
  
            # parameterize phi prior p(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            p_phi_t = self._parameterize_prior_phi(p_alpha_sample, h_p_phi_t)
            # parameterize phi posterior q(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            q_phi_t = self._parameterize_posterior_phi(q_alpha_sample, h_q_phi_t)
            # calculate kl divergence KL(q(phi_t|phi_t-1)||p(phi_t|phi_t-1))
            # (batch_size, num_nodes, phi_dim)
            kl_phi += [kl_divergence(q_phi_t, p_phi_t)] 
     
            # sample phi prior
            # (batch_size, num_nodes, phi_dim)
            p_phi_sample_t = p_phi_t.mean
            # sample phi posterior
            # (batch_size, num_nodes, phi_dim)
            q_phi_sample_t = q_phi_t.rsample() if self.training else q_phi_t.mean
            # update hidden state phi prior
            # (1, num_nodes * batch_size, phi_dim)
            h_q_phi_t = q_phi_sample_t.reshape(1, -1, self.phi_dim).contiguous() 
            # update hidden state phi posterior
            # (1, num_nodes * batch_size, phi_dim)
            h_p_phi_t = p_phi_sample_t.reshape(1, -1, self.phi_dim).contiguous() 
         
            # forward pass on edges at a given timepoint
            # (batch_size, num_windows, num_edges, 2) -> (batch_size, num_edges, 2)
            # (batch_size, num_windows, num_edges) -> (batch_size, num_edges)
            edges_t, weights_t = edges[:, t, ...], weights[:, t, ...]
            nll_t, kl_z_t = self._forward_edges(edges_t, weights_t, p_phi_sample_t, q_phi_sample_t, q_beta_sample_t)
            
            nll += [nll_t]
            kl_z += [kl_z_t]

            if self.inference:
                self.q_phi += [q_phi_t]
                self.q_beta += [q_beta_t]
        
        # stack results along temporal dimension
        nll, kl_beta = torch.stack(nll, dim=-1), torch.stack(kl_beta, dim=-1)
        kl_phi, kl_z = torch.stack(kl_phi, dim=-1), torch.stack(kl_z, dim=-1)

        return nll, kl_beta, kl_phi, kl_z

    def forward(self, X, idx=None):
        assert ~(self.training and self.inference)
        if self.inference:
            self.q_beta = []
            self.q_phi = []

        # check dimension of input to ensure multivariate timeseries
        if X.ndim != 3:
            raise ValueError("input must have shape (batch_size, num_nodes, time_len)")
        
        # dynamic connectivity matrix from timeseries
        # (batch_size, num_nodes num_timepoints) -> (batch_size, num_windows, num_nodes, num_nodes)
        A = self._dynamic_connectivity_matrix(X)
        
        # edge weights from dynamic connectivity matrix
        # (batch_size, num_windows, num_edges, 2), (batch_size, num_windows, num_edges)
        edges, weights = self._to_edge_weights(A)
     
        # parameterize alpha prior
        p_alpha = self._parameterize_prior_alpha(idx)
        # parameterize alpha posterior
        q_alpha = self._parameterize_posterior_alpha(idx)
        # sample prior alpha
        # (batch_size, alpha_dim)
        p_alpha_sample = p_alpha.mean
        # sample posterior alpha
        # if not training just sample from the prior
        # (batch_size, alpha_dim)
        q_alpha_sample = q_alpha.rsample() if not self.training else p_alpha_sample
        # calculate kl divergence KL(q(alpha|h)||p(alpha))
        # (batch_size, alpha_dim)
        kl_alpha = kl_divergence(q_alpha, p_alpha)
  
        # forward pass graph timeseries
        nll, kl_beta, kl_phi, kl_z = self._forward_timesteps(edges, weights, p_alpha_sample, q_alpha_sample)
        
        return nll, kl_alpha, kl_beta, kl_phi, kl_z


    # def get_embeddings(self, x):
    #     self.training = False
    #     self.inference = True

    #     self.q_beta = []
    #     self.q_phi = []

    #     with torch.no_grad():
    #         _, _, _, _, _ = self.forward(x)

    #     # subject embeddings
    #     alpha = self._parameterize_posterior_alpha(None).mean
        
    #     # community embeddings
    #     # (batch_size, num_communities, beta_dim, num_windows)
    #     beta = torch.stack([q.mean for q in self.q_beta], dim=-1)
    #     # node embeddings
    #     # (batch_size, num_nodes, phi_dim, num_windows)
    #     phi = torch.stack([q.mean for q in self.q_phi], dim=-1)
        
    #     return dict(alpha=alpha, beta=beta, phi=phi)
       
    # def get_community_node_dist(self, x):
    #     beta = self.get_embeddings(x)["beta"]
    #     # (batch_size, num_communities, beta_dim, num_windows) -> (batch_size * num_windows, num_communities, beta_dim)
    #     beta = beta.permute(0, 3, 1, 2).reshape(-1, self.num_communities, self.beta_dim)
    #     # (batch_size * num_windows, num_communities, beta_dim) -> (batch_size * num_windows, num_communities, num_nodes)
    #     node_communities = self.fc_p_c(beta)
    #     # (batch_size * num_windows, num_communities, num_nodes) -> (batch_size, num_communities, num_nodes, num_windows)
    #     return node_communities.reshape(x.shape[0], -1, self.num_communities, self.num_nodes).permute(0, 2, 3, 1)






# threshold = torch.tensor(np.percentile(A.cpu().numpy(), percentile, axis=-1)).unsqueeze(-1).to(self.device)
    # A[A <= threshold] = 0.
    # _A = A.reshape(batch_size, -1, self.num_nodes, self.num_nodes)
    # print(_A.shape)
    # for idx1, _A in enumerate(_A):
    #     for idx2, __A in enumerate(_A):
    #         pass
            # print(idx1, idx2, __A.nonzero().shape)
            # is_not_nan_inf(A)
