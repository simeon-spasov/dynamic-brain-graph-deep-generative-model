import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, RelaxedOneHotCategorical
from torch.distributions.utils import clamp_probs
from torch.distributions.kl import _kl_normal_normal as kl_normal


class Accumulator:
   
    def __init__(self):
        self.data = dict(nll_pos=list(), nll_neg=list(), kl_beta=list(), kl_phi=list(), kl_z=list())

    def append(self, *args):
        for k, v in zip(self.data.keys(), args):
            self.data[k] += [v]

    def stack(self, dim=-1):
        self.data= {k: torch.stack(self.data[k], dim=dim) for k in self.data.keys()}


def kl_categorical(q, p):
    kl = clamp_probs(q.probs) * (q.logits - p.logits)
    return kl


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.reshape(views).expand(expanse)
    return torch.gather(input, dim, index)


def init_weights(model):
    for module in model.modules():

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -1.0, 1.0)

        if isinstance(module, nn.GRU):
            pass


class DynamicGraph(nn.Module):
    def __init__(self, num_samples, num_nodes, num_communities, num_timesteps, criterion,
                alpha_dim=14, beta_dim=17, phi_dim=23, alpha_std=0.1, tau=1., device=torch.device("cpu")):
        super(DynamicGraph, self).__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.num_timesteps = num_timesteps

        self.criterion = criterion

        self.alpha_dim = alpha_dim
        self.alpha_std = alpha_std
        self.beta_dim = beta_dim
        self.phi_dim = phi_dim
        self.tau = tau
        
        self.warmup = False

        self._init_params()
        self.to(device)


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
    

    def _init_params(self):
        self._create_alpha_prior_params()
        self._create_beta_prior_params()
        self._create_phi_prior_params()
        self._create_z_prior_params()
        self._create_c_likelihood_params()
        self._create_alpha_posterior_params()
        self._create_beta_posterior_params()
        self._create_phi_posterior_params()
        self._create_z_posterior_params()
        self.apply(init_weights)


    def _parameterise_prior_alpha(self, batch_size):
        mean = self.p_alpha_mean
        std = self.p_alpha_std
        return Normal(mean, std).expand(torch.Size([batch_size, self.alpha_dim]))

    
    def _parameterise_posterior_alpha(self, sample_idx):
        h = self.h_q_alpha(sample_idx).squeeze(1)
        out = self.fc_q_alpha(h)
        mean = out[..., :self.alpha_dim]
        std = F.softplus(out[..., self.alpha_dim:])
        return Normal(mean, std)


    def _parameterise_prior_beta(self, alpha, h):
        out = self.fc_p_alpha_beta(alpha)
        out = out.reshape(-1, 1, self.beta_dim)
        _, out = self.rnn_p_beta(out, h)
        out = out.reshape(-1, self.num_communities, self.beta_dim)
        out = self.fc_p_beta(out)
        mean = out[..., :self.beta_dim]
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)


    def _parameterise_posterior_beta(self, alpha, h):
        out = self.fc_q_alpha_beta(alpha)
        out = out.reshape(-1, 1, self.beta_dim)
        _, out = self.rnn_q_beta(out, h)
        out = out.reshape(-1, self.num_communities, self.beta_dim)
        out = self.fc_q_beta(out)
        mean = out[..., :self.beta_dim]
        std = F.softplus(out[..., self.beta_dim:])
        return Normal(mean, std)


    def _parameterise_prior_phi(self, alpha, h):
        out = self.fc_p_alpha_phi(alpha)
        out = out.reshape(-1, 1, self.phi_dim)
        _, out = self.rnn_p_phi(out, h)
        out = out.reshape(-1, self.num_nodes, self.phi_dim)
        out = self.fc_p_phi(out)
        mean = out[..., :self.phi_dim]
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)


    def _parameterise_posterior_phi(self, alpha, h):
        out = self.fc_q_alpha_phi(alpha)
        out = out.reshape(-1, 1, self.phi_dim)
        _, out = self.rnn_q_phi(out, h)
        out = out.reshape(-1, self.num_nodes, self.phi_dim)
        out = self.fc_q_phi(out)
        mean = out[..., :self.phi_dim]
        std = F.softplus(out[..., self.phi_dim:])
        return Normal(mean, std)


    def _parameterise_prior_z(self, p_phi_sample, w):
        phi_w = batched_index_select(p_phi_sample, 1, w)
        logits = self.fc_p_z(phi_w)
        return RelaxedOneHotCategorical(logits=logits, temperature=self.tau)


    def _parameterise_posterior_z(self, q_phi_sample, w, c):
        phi_w = batched_index_select(q_phi_sample, 1, w)
        phi_c = batched_index_select(q_phi_sample, 1, c)
        h = torch.cat((phi_w, phi_c), dim=-1)
        logits = self.fc_q_z(h)
        return RelaxedOneHotCategorical(logits=logits, temperature=self.tau)
  

    def _parameterise_likelihood_c(self, beta, z):
        beta = beta.unsqueeze(1)
        z = z.unsqueeze(dim=-1)
        beta_z = beta * z
        beta_z = beta_z.reshape(*list(beta_z.shape[0:2]), -1)
        logits = self.fc_p_c(beta_z)
        return Categorical(logits=logits)

    
    def _forward_graph_edges(self, edges, edge_weights, p_phi_sample, q_phi_sample, q_beta_sample):
        # unpack edges from and edges to
        w, c = edges[..., 0], edges[..., 1]

        # parameterise z prior p(z | w)
        p_z = self._parameterise_prior_z(p_phi_sample, w)
        # parameterise z posterior q(z | w, c)
        q_z = self._parameterise_posterior_z(q_phi_sample, w, c)  
        # kl divergence z
        kl_z = kl_categorical(q_z, p_z)
     
        # sample z posterior q(z | w, c)
        q_z_sample = q_z.rsample() if self.training else F.one_hot(q_z.logits.argmax(-1), num_classes=self.num_communities)  
        # parameterise c likelihood p(c | z)
        p_c = self._parameterise_likelihood_c(q_beta_sample, q_z_sample)
 
        # negative log likelihood = multiclass cross-entropy
        nll = F.cross_entropy(p_c.logits.reshape(-1, self.num_nodes), c.reshape(-1), reduction="none")
        nll = nll.reshape(q_z_sample.shape[0], -1)
        
        return nll, kl_z
    

    def _forward_graph_snapshots(self, pos_edges, pos_edge_weights, neg_edges, p_alpha_sample, q_alpha_sample):
    
        neg_edge_weights = torch.zeros_like(pos_edge_weights)
        
        batch_size = pos_edges.shape[0]
        h_p_beta = self.h_p_beta.repeat_interleave(batch_size, dim=1) 
        h_p_phi = self.h_p_phi.repeat_interleave(batch_size, dim=1) 
        h_q_beta = self.h_q_beta.repeat_interleave(batch_size, dim=1) 
        h_q_phi = self.h_q_phi.repeat_interleave(batch_size, dim=1) 

        metrics = Accumulator()
        
        for t in range(self.num_timesteps):

            pos_e, pos_ew = pos_edges[:, t, ...], pos_edge_weights[:, t, ...]
            neg_e, neg_ew = neg_edges[:, t, ...], neg_edge_weights[:, t, ...]

            # parameterise prior p(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            p_beta = self._parameterise_prior_beta(p_alpha_sample, h_p_beta)
            # parameterise posterior q(beta_t|beta_t-1) = GRU(alpha, h_t-1)
            q_beta = self._parameterise_posterior_beta(q_alpha_sample, h_q_beta)
            # calculate kl divergence KL(q(beta_t|beta_t-1)||p(beta_t|beta_t-1))
            kl_beta = kl_normal(q_beta, p_beta)

            # sample prior beta_t ~ p(beta_t|beta_t-1) 
            p_beta_sample = p_beta.mean  
            # sample posterior beta_t ~ q(beta_t|beta_t-1) 
            q_beta_sample = q_beta.rsample()

            # update hidden state beta prior h_t = beta_t
            h_p_beta = p_beta_sample.reshape(1, -1, self.beta_dim) 
            # update hidden state beta posterior h_t = beta_t
            h_q_beta = q_beta_sample.reshape(1, -1, self.beta_dim) 

            # parameterise phi prior p(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            p_phi = self._parameterise_prior_phi(p_alpha_sample, h_p_phi)
            # parameterise phi posterior q(phi_t|phi_t-1) = GRU(alpha, h_t-1)
            q_phi = self._parameterise_posterior_phi(q_alpha_sample, h_q_phi)
            # calculate kl divergence KL(q(phi_t|phi_t-1)||p(phi_t|phi_t-1))
            kl_phi = kl_normal(q_phi, p_phi)

            # sample phi prior
            p_phi_sample = p_phi.mean
            # sample phi posterior
            q_phi_sample = q_phi.rsample()
            # update hidden state phi prior
            h_q_phi = q_phi_sample.reshape(1, -1, self.phi_dim)
            # update hidden state phi posterior
            h_p_phi = p_phi_sample.reshape(1, -1, self.phi_dim) 
            
            nll_pos, kl_z = self._forward_graph_edges(pos_e, pos_ew, p_phi_sample, q_phi_sample, q_beta_sample)
            nll_neg, _ = self._forward_graph_edges(neg_e, neg_ew, p_phi_sample, q_phi_sample, q_beta_sample)
            
            metrics.append(nll_pos, nll_neg, kl_beta, kl_phi, kl_z)
        
        metrics.stack()

        return metrics.data


    def freeze(self, mode=True):
        if mode:
            for name, param in self.named_parameters():
                param.requires_grad = True if name == "h_q_alpha.weight" else False
        else:
            for name, param in self.named_parameters():
                param.requires_grad = True


    def forward(self, subject_idx, pos_edges, pos_edge_weights, neg_edges):
  
        # parameterise alpha prior
        p_alpha = self._parameterise_prior_alpha(subject_idx.shape[0])
        # parameterise alpha posterior
        q_alpha = self._parameterise_posterior_alpha(subject_idx)

        kl_alpha = kl_normal(q_alpha, p_alpha)

        # sample prior alpha
        p_alpha_sample = p_alpha.mean

        # sample posterior alpha
        q_alpha_sample = q_alpha.rsample()
    
        metrics = self._forward_graph_snapshots(pos_edges, pos_edge_weights, neg_edges, p_alpha_sample, q_alpha_sample)
        loss, loss_parts = self.criterion(**metrics, kl_alpha=kl_alpha, warmup=self.warmup)

        return loss, loss_parts 