import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def _gumbel_softmax(logits, device, tau=1, hard=False, eps=1e-10, dim=-1):
    shape = logits.size()
    U = torch.rand(shape).to(device)  # sample from uniform [0, 1)
    g = -torch.log(-torch.log(U + eps) + eps)
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Model(nn.Module):
    def __init__(self, num_samples, num_nodes,
                 embedding_dim=32,
                 categorical_dim=3,
                 gamma=0.1,
                 sigma=1.,
                 device=torch.device("cpu")):

        super().__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.gamma = gamma
        self.sigma = sigma
        self.device = device

        self.beta_mean = nn.Linear(self.embedding_dim, embedding_dim)
        self.beta_std = nn.Sequential(nn.Linear(self.embedding_dim, embedding_dim), nn.Softplus())
        self.phi_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.phi_std = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Softplus())
        self.nn_pi = nn.Linear(self.embedding_dim, self.categorical_dim)

        self.rnn_nodes = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)
        self.rnn_comms = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)

        self.alpha_mean = nn.Embedding(self.num_samples, self.embedding_dim)
        self.alpha_std = nn.Embedding(self.num_samples, self.embedding_dim)

        self.subject_to_phi = nn.Linear(self.embedding_dim, self.num_nodes * self.embedding_dim)
        self.subject_to_beta = nn.Linear(self.embedding_dim, self.categorical_dim * self.embedding_dim)

        self.alpha_mean_prior = torch.zeros(self.embedding_dim)
        self.alpha_std_scalar = 1.

        self.decoder = nn.Sequential(nn.Linear(embedding_dim, num_nodes))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, batch_data, temp=1.):
        loss = 0
        for data in batch_data:
            subject_idx, dynamic_graph, _ = data
            loss += self._forward(subject_idx, dynamic_graph, temp)
        return loss

    def _forward(self, subject_idx, dynamic_graph, temp):

        logging.debug(f"Forward pass for subject idx: {subject_idx}")

        loss = 0
        # sample subject embedding from posterior
        alpha_mean_n = self.alpha_mean.weight[subject_idx]
        alpha_std_n = F.softplus(self.alpha_std.weight[subject_idx])
        alpha_n = self._reparameterized_sample(alpha_mean_n, alpha_std_n)

        KLD_alpha = self._kld_gauss(alpha_mean_n, alpha_std_n,
                                    self.alpha_mean_prior.to(self.device),
                                    self.alpha_std_scalar
                                    )

        loss += 0 * KLD_alpha  # For debugging simplicity. TODO: need to figure out the weights for these KL terms.

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.
        # phi_0_mean = torch.zeros((self.num_nodes, self.embedding_dim)).to(self.device)
        # beta_0_mean = torch.zeros((self.categorical_dim, self.embedding_dim)).to(self.device)
        logging.debug(f'alpha n is {alpha_n}')
        logging.debug(f'subject_to_phi is {self.subject_to_phi}')
        logging.debug(f'subject_to_phi weights is {self.subject_to_phi.weight}')
        logging.debug(f'subject_to_phi bias is {self.subject_to_phi.bias}')
        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)

        # Initialize the priors over nodes (phi) and communities (beta)
        phi_prior_mean = phi_0_mean
        beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        # iterate all over all edges in a graph at a single time point
        for snapshot_idx, graph in enumerate(dynamic_graph):

            train_edges = [(u, v) for u, v in graph.edges()]
            if self.training:
                np.random.shuffle(train_edges)

            batch = torch.LongTensor(train_edges).to(self.device)
            assert batch.shape == (len(train_edges), 2)
            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))

            # Update node and community hidden states of GRUs
            # nodes_in = torch.cat([alpha_n.unsqueeze(0).expand(self.num_nodes, -1),
            #                     phi_prior_mean], dim = -1)
            nodes_in = torch.cat([phi_prior_mean, phi_prior_mean], dim=-1)

            nodes_in = nodes_in.view(1, self.num_nodes, 2 * self.embedding_dim)

            _, h_phi = self.rnn_nodes(nodes_in, h_phi)

            # comms_in = torch.cat([alpha_n.unsqueeze(0).expand(self.categorical_dim, -1),
            #                      beta_prior_mean], dim = -1)
            comms_in = torch.cat([beta_prior_mean, beta_prior_mean], dim=-1)

            comms_in = comms_in.view(1, self.categorical_dim, 2 * self.embedding_dim)

            _, h_beta = self.rnn_comms(comms_in, h_beta)

            # Produce node and community mean and std of respective posteriors
            beta_mean_t = self.beta_mean(h_beta[-1])
            beta_std_t = self.beta_std(h_beta[-1])

            phi_mean_t = self.phi_mean(h_phi[-1])
            phi_std_t = self.phi_std(h_phi[-1])

            # Sample node and community representations
            beta_sample = self._reparameterized_sample(beta_mean_t, beta_std_t)
            phi_sample = self._reparameterized_sample(phi_mean_t, phi_std_t)

            recon, posterior_z, prior_z = self._edge_reconstruction(w, c, phi_sample, beta_sample, temp)

            # per subject loss for time t
            KLD_z, BCE = self._vGraph_loss(recon, posterior_z, prior_z, c)
            KLD_beta = self._kld_gauss(beta_mean_t, beta_std_t, beta_prior_mean, self.gamma)
            KLD_phi = self._kld_gauss(phi_mean_t, phi_std_t, phi_prior_mean, self.sigma)

            logging.debug(f"For Subject index: {subject_idx}.\n"
                          f" Snapshot index: {snapshot_idx}: "
                          f"BCE loss is {BCE}. KLD_z loss is {KLD_z}.")

            loss_edges = (KLD_z + BCE) / c.shape[0]
            loss += loss_edges + 0 * KLD_beta / self.categorical_dim + 0 * KLD_phi / self.num_nodes

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample

        return loss

    def _edge_reconstruction(self, w, c, phi_sample, beta_sample, temp):
        q = self.nn_pi(phi_sample[w] * phi_sample[c])

        if self.training:
            z = _gumbel_softmax(q, self.device, tau=temp, hard=True)
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            src = torch.ones_like(tmp).float()
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, src)

        q_prior = self.nn_pi(phi_sample[w])
        prior_z = F.softmax(q_prior, dim=-1)
        # Sample community assignment from posterior q(z|w,c)
        new_z = torch.mm(z, beta_sample)
        # recon gives distribution over the nodes p(c|z)
        recon = self.decoder(new_z)

        return recon, F.softmax(q, dim=-1), prior_z

    def inference(self, subject_graphs):
        subjects = {}
        for subject in subject_graphs:
            subject_idx, subject_graphs = subject
            node_distrib_over_communities = self._inference(subject_idx, subject_graphs)
            subjects[subject_idx] = node_distrib_over_communities

        return subjects

    def _inference(self, subject_idx, batch_graphs):
        alpha_n = self.alpha_mean.weight[subject_idx].to(self.device)

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.
        # phi_0_mean = torch.zeros((self.num_nodes, self.embedding_dim)).to(self.device)
        # beta_0_mean = torch.zeros((self.categorical_dim, self.embedding_dim)).to(self.device)

        # Initialize the priors over nodes (phi) and communities (beta)
        # phi_prior_mean = phi_0_mean
        # beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        # h_beta = torch.zeros(1, self.categorical_dim,
        #                     self.embedding_dim).to(self.device)
        # h_phi = torch.zeros(1, self.num_nodes,
        #                    self.embedding_dim).to(self.device)

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.
        # phi_0_mean = torch.zeros((self.num_nodes, self.embedding_dim)).to(self.device)
        # beta_0_mean = torch.zeros((self.categorical_dim, self.embedding_dim)).to(self.device)
        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)

        # Initialize the priors over nodes (phi) and communities (beta)
        phi_prior_mean = phi_0_mean
        beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        h_beta = torch.zeros(1, self.categorical_dim,
                             self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes,
                            self.embedding_dim).to(self.device)

        node_distrib_over_communities = []
        for i, graph in enumerate(batch_graphs):
            nodes_in = torch.cat([phi_prior_mean,
                                  phi_prior_mean], dim=-1)

            nodes_in = nodes_in.view(1, self.num_nodes, 2 * self.embedding_dim)

            _, h_phi = self.rnn_nodes(nodes_in, h_phi)

            # comms_in = torch.cat([alpha_n.unsqueeze(0).expand(self.categorical_dim, -1),
            #                      beta_prior_mean], dim = -1)
            comms_in = torch.cat([beta_prior_mean,
                                  beta_prior_mean], dim=-1)

            comms_in = comms_in.view(1, self.categorical_dim, 2 * self.embedding_dim)

            _, h_beta = self.rnn_comms(comms_in, h_beta)

            # Sample node and community representations
            beta_sample = self.beta_mean(h_beta[-1])
            phi_sample = self.phi_mean(h_phi[-1])

            node_distrib_over_communities_t = self.decoder(beta_sample)
            node_distrib_over_communities_t = F.softmax(
                node_distrib_over_communities_t,
                dim=-1).cpu().detach().data.numpy()
            node_distrib_over_communities.append(
                node_distrib_over_communities_t)

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample

        return node_distrib_over_communities

    def _reparameterized_sample(self, mean, std):
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean)

    def _vGraph_loss(self, recon_c, q_y, prior_z, c):
        recon_c_softmax = F.log_softmax(recon_c, dim=-1)
        BCE = F.nll_loss(recon_c_softmax, c, reduction='none')
        BCE = (BCE).sum()
        log_qy = torch.log(q_y + 1e-20)
        KLD = (
            torch.sum(q_y *
                      (log_qy - torch.log(prior_z + 1e-20)), dim=-1)).sum()
        return KLD, BCE

    def _kld_gauss(self, mu_1, std_1, mu_2, std_2_scale):
        std_2 = Variable(torch.ones_like(std_1)).mul(std_2_scale).to(
            self.device)
        KLD = 0.5 * torch.sum(
            (2 * torch.log(std_2 + 1e-20) - 2 * torch.log(std_1 + 1e-20) +
             (std_1.pow(2) + (mu_1 - mu_2).pow(2)) / std_2.pow(2) - 1))
        return KLD
