import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import (Categorical, Normal, RelaxedOneHotCategorical,
                                 kl_divergence)

from src.connectivity import ConnectivityMeasure
from src.utils import dense_to_edge_weights


class Model(nn.Module):
    def __init__(self, num_samples, num_nodes, num_communities, num_timesteps,
                 alpha_dim=14, beta_dim=17, phi_dim=23, alpha_std=0.1, temp=1.,
                 window_size=10, window_stride=2, measure="correlation"):
        super().__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.num_timesteps = num_timesteps

        self.alpha_dim = alpha_dim
        self.alpha_std = alpha_std
        self.beta_dim = beta_dim
        self.phi_dim = phi_dim
        self.temp = temp

        self.window_size = window_size
        self.window_stride = window_stride
        self.connectivity_measure = ConnectivityMeasure(measure)
        
        self._initialize_params()

    def _create_alpha_prior_params(self):
        self.p_alpha_mean = torch.zeros(self.alpha_dim)
        self.p_alpha_std = torch.ones(self.alpha_dim) * self.alpha_std

    def _create_alpha_posterior_params(self):    
        self.h_q_alpha = nn.Embedding(self.num_samples, self.alpha_dim, padding_idx=0)
        self.fc_q_alpha = nn.Linear(self.alpha_dim, 2 * self.alpha_dim, bias=True)
            
    def _create_beta_prior_params(self):
        self.fc_p_alpha_beta = nn.Linear(self.alpha_dim, self.beta_dim * self.num_communities, bias=True)
        self.h_p_beta = torch.zeros(1, self.num_communities, self.beta_dim)
        self.rnn_p_beta = nn.GRU(self.beta_dim, self.beta_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_p_beta = nn.Linear(self.beta_dim, 2 * self.beta_dim, bias=True)

    def _create_beta_posterior_params(self):
        self.fc_q_alpha_beta = nn.Linear(self.alpha_dim, self.beta_dim * self.num_communities, bias=True)
        self.h_q_beta = torch.zeros(1, self.num_communities, self.beta_dim)
        self.rnn_q_beta = nn.GRU(self.beta_dim, self.beta_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_q_beta = nn.Linear(self.beta_dim, 2 * self.beta_dim, bias=True)

    def _create_phi_prior_params(self):
        self.fc_p_alpha_phi = nn.Linear(self.alpha_dim, self.phi_dim * self.num_nodes, bias=True)
        self.h_p_phi = torch.zeros(1, self.num_nodes, self.phi_dim)
        self.rnn_p_phi = nn.GRU(self.phi_dim, self.phi_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_p_phi = nn.Linear(self.phi_dim, 2 * self.phi_dim, bias=True)
        
    def _create_phi_posterior_params(self):
        self.fc_q_alpha_phi = nn.Linear(self.alpha_dim, self.phi_dim * self.num_nodes, bias=True)
        self.h_q_phi = torch.zeros(1, self.num_nodes, self.phi_dim)
        self.rnn_q_phi = nn.GRU(self.phi_dim, self.phi_dim, num_layers=1, batch_first=True, bias=True)
        self.fc_q_phi = nn.Linear(self.phi_dim, 2 * self.phi_dim, bias=True)

    def _create_z_prior_params(self):
        self.fc_p_z = nn.Linear(self.phi_dim, self.num_communities, bias=True)
        
    def _create_z_posterior_params(self):
        self.fc_q_z = nn.Linear(2 * self.phi_dim, self.num_communities, bias=True)

    def _create_c_likelihood_params(self):
        self.fc_p_c = nn.Linear(self.beta_dim * self.num_communities, self.num_nodes, bias=True)
    
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
    # p(alpha) = Normal(mean, std)
    def _parameterize_prior_alpha(self, sample_idx):
        batch_size = sample_idx.shape[0]
        mean = self.p_alpha_mean
        std = self.p_alpha_std
        return Normal(mean, std).expand(torch.Size([batch_size, self.alpha_dim]))

    # posterior over subject embeddings -> parameterize from ebeddings
    # q(alpha|h) = Normal((mean, std) = NN(h)))
    def _parameterize_posterior_alpha(self, sample_idx):
        h = self.h_q_alpha(sample_idx).squeeze(1)
        out = self.fc_q_alpha(h)
        mean = out[..., :self.alpha_dim]
        std = F.softplus(out[..., self.alpha_dim:])
        return Normal(mean, std)

    # prior over community embeddings
    # p(beta) = 
    def _parameterize_prior_beta(self, alpha, h):
        out = self.fc_p_alpha_beta(alpha)
        out = out.reshape(-1, 1, self.beta_dim)
        _, out = self.rnn_p_beta(out, h)
        out = out.reshape(-1, self.num_communities, self.beta_dim)
        out = self.fc_p_beta(out)
        mean = out[..., :self.beta_dim]
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)

    # posterior over community embeddings
    # q(beta) = 
    def _parameterize_posterior_beta(self, alpha, h):
        out = self.fc_q_alpha_beta(alpha)
        out = out.reshape(-1, 1, self.beta_dim)
        _, out = self.rnn_q_beta(out, h)
        out = out.reshape(-1, self.num_communities, self.beta_dim)
        out = self.fc_q_beta(out)
        mean = out[..., :self.beta_dim]
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)

    def _parameterize_prior_phi(self, alpha, h):
        out = self.fc_p_alpha_phi(alpha)
        out = out.reshape(-1, 1, self.phi_dim)
        _, out = self.rnn_p_phi(out, h)
        out = out.reshape(-1, self.num_nodes, self.phi_dim)
        out = self.fc_p_phi(out)
        mean = out[..., :self.phi_dim]
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)

    def _parameterize_posterior_phi(self, alpha, h):
        out = self.fc_q_alpha_phi(alpha)
        out = out.reshape(-1, 1, self.phi_dim)
        _, out = self.rnn_q_phi(out, h)
        out = out.reshape(-1, self.num_nodes, self.phi_dim)
        out = self.fc_q_phi(out)
        mean = out[..., :self.phi_dim]
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)

    def _parameterize_prior_z(self, p_phi_sample, w):
        phi_w = batched_index_select(p_phi_sample, 1, w)
        logits = self.fc_p_z(phi_w)
        return Categorical(logits=logits)

    def _parameterize_posterior_z(self, q_phi_sample, w, c):
        phi_w = batched_index_select(q_phi_sample, 1, w)
        phi_c = batched_index_select(q_phi_sample, 1, c)
        h = torch.cat((phi_w, phi_c), dim=-1)
        logits = self.fc_q_z(h)
        return Categorical(logits=logits)
  
    def _parameterize_likelihood_c(self, beta, z):
        beta = beta.unsqueeze(1)
        z = z.unsqueeze(dim=-1)
        beta_z = beta * z
        beta_z = beta_z.reshape(*list(beta_z.shape[0:2]), -1)
        logits = self.fc_p_c(beta_z)
        return Categorical(logits=logits)

    def _forward_edges(self, edges, edge_weights, p_phi_sample, q_phi_sample, q_beta_sample):
        # unpack edges (w, c)
        # (batch_size, num_edges, 2) -> (batch_size, num_edges), (batch_size, num_edges)
        w, c = edges[..., 0], edges[..., 1]
        # parameterize prior p(z | w)
        p_z = self._parameterize_prior_z(p_phi_sample, w)
        # parameterize posterior q(z | w, c)
        q_z = self._parameterize_posterior_z(q_phi_sample, w, c)  
        # kl divergence
        kl_z = kl_divergence(q_z, p_z)
        # sample posterior q(z | w, c)
        q_z_sample = q_z.rsample() if self.training else F.one_hot(q_z.logits.argmax(-1), num_classes=self.num_communities)  
        # parameterize likelihood p(c | z)
        p_c = self._parameterize_likelihood_c(q_beta_sample, q_z_sample)
        # negative log likelihood = cross-entropy
        nll = F.cross_entropy(p_c.logits.reshape(-1, self.num_nodes), c.reshape(-1), reduction="none")
        nll = nll.reshape(q_z_sample.shape[0], -1)
        return nll, kl_z
    
    def _forward_graph(self, edges, edge_weights, p_alpha_sample, q_alpha_sample):
       
        batch_size, num_windows = edges.shape[:2]
        h_p_beta = self.h_p_beta.repeat_interleave(batch_size, dim=1) 
        h_p_phi = self.h_p_phi.repeat_interleave(batch_size, dim=1) 
        h_q_beta = self.h_q_beta.repeat_interleave(batch_size, dim=1) 
        h_q_phi = self.h_q_phi.repeat_interleave(batch_size, dim=1) 

        nll, kl_beta, kl_phi, kl_z = [], [], [], []
        # loop over time
        for t in range(num_windows):
            # parameterize prior p(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            p_beta_t = self._parameterize_prior_beta(p_alpha_sample, h_p_beta)
            # parameterize posterior q(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            q_beta_t = self._parameterize_posterior_beta(q_alpha_sample, h_q_beta)
            # calculate kl divergence KL(q(beta_t|beta_t-1)||p(beta_t|beta_t-1))
            kl_beta += [kl_divergence(q_beta_t, p_beta_t)]

            # sample prior beta_t ~ p(beta_t|beta_t-1) 
            p_beta_sample_t = p_beta_t.mean  
            # sample posterior beta_t ~ q(beta_t|beta_t-1) 
            q_beta_sample_t = q_beta_t.rsample()

            # update hidden state beta prior h_t = beta_t
            h_p_beta_t = p_beta_sample_t.reshape(1, -1, self.beta_dim) 
            # update hidden state beta posterior h_t = beta_t
            h_q_beta_t = q_beta_sample_t.reshape(1, -1, self.beta_dim) 

            # parameterize phi prior p(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            p_phi_t = self._parameterize_prior_phi(p_alpha_sample, h_p_phi)
            # parameterize phi posterior q(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            q_phi_t = self._parameterize_posterior_phi(q_alpha_sample, h_q_phi)
            # calculate kl divergence KL(q(phi_t|phi_t-1)||p(phi_t|phi_t-1))
            kl_phi += [kl_divergence(q_phi_t, p_phi_t)] 

            # sample phi prior
            p_phi_sample_t = p_phi_t.mean
            # sample phi posterior
            q_phi_sample_t = q_phi_t.rsample()
            # update hidden state phi prior
            h_q_phi_t = q_phi_sample_t.reshape(1, -1, self.phi_dim)
            # update hidden state phi posterior
            h_p_phi_t = p_phi_sample_t.reshape(1, -1, self.phi_dim) 

            # forward pass on edges and weights at a given timepoint
            # (batch_size, num_windows, num_edges, 2) -> (batch_size, num_edges, 2)
            # (batch_size, num_windows, num_edges) -> (batch_size, num_edges)
            edges_t, edge_weights_t = edges[:, t, ...], edge_weights[:, t, ...]
            nll_t, kl_z_t = self._forward_edges(edges_t, edge_weights_t, p_phi_sample_t, q_phi_sample_t, q_beta_sample_t)
            nll += [nll_t]
            kl_z += [kl_z_t]

        # stack results along temporal dimension
        nll, kl_beta = torch.stack(nll, dim=-1), torch.stack(kl_beta, dim=-1)
        kl_phi, kl_z = torch.stack(kl_phi, dim=-1), torch.stack(kl_z, dim=-1)
        
        return nll, kl_beta, kl_phi, kl_z

    def _compute_dynamic_connectivity(self, x):
        batch_size = x.shape[0]
        # padding
        if (self.num_timesteps % self.window_stride == 0):
            pad = max(self.window_size - self.window_stride, 0)
        else:
            pad = max(self.window_size - (self.num_timesteps % self.window_stride), 0)
        pad_left = pad // 2
        pad_right = pad - pad_left
        x = F.pad(x, (pad_left, pad_right))
        # split into windows
        # (batch_size, num_nodes, num_timesteps) -> (batch_size, num_nodes, num_windows, window_size)
        x = x.unfold(-1, self.window_size, self.window_stride)
        # (batch_size, num_nodes, num_windows, window_size) -> (batch_size * num_windows, num_nodes, window_size)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.num_nodes, self.window_size)
        
        # compute connectivity between nodes in each window
        # (batch_size * num_windows, num_nodes, window_size) -> (batch_size * num_windows, num_nodes, num_nodes)
        A = self.connectivity_measure(x)

        # (batch_size * num_windows, num_nodes, num_nodes) -> (batch_size, num_windows, num_nodes, num_nodes)
        return A.reshape(batch_size, -1, self.num_nodes, self.num_nodes)
  
    def forward(self, idx, x):
        # check dimension of input to ensure multivariate timeseries
        if x.ndim != 3:
            raise ValueError("input must have shape (batch_size, num_nodes, time_len)")
        
        # dynamic connectivity matrix from timeseries
        # (batch_size, num_nodes num_timepoints) -> (batch_size, num_windows, num_nodes, num_nodes)
        dyn_conn_matrix = self._compute_dynamic_connectivity(x)
        # edges and weights from dynamic connectivity matrix
        # (batch_size, num_windows, num_edges, 2), (batch_size, num_windows, num_edges)
        edges, edge_weights = dense_to_edge_weights(dyn_conn_matrix)

        # parameterize alpha prior
        p_alpha = self._parameterize_prior_alpha(idx)
        # parameterize alpha posterior
        q_alpha = self._parameterize_posterior_alpha(idx)
        # sample prior alpha
        p_alpha_sample = p_alpha.mean
        # sample posterior alpha
        # if not training just sample from the prior
        q_alpha_sample = q_alpha.rsample() if self.training else p_alpha.sample()
        # calculate kl divergence KL(q(alpha|h)||p(alpha))
        kl_alpha = kl_divergence(q_alpha, p_alpha)
 
        # forward pass graph timeseries
        nll, kl_beta, kl_phi, kl_z = self._forward_graph(edges, edge_weights, p_alpha_sample, q_alpha_sample)

        return nll, kl_alpha, kl_beta, kl_phi, kl_z
