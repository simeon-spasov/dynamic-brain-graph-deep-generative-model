import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.autograd import Variable

from .utils import sample_pos_neg_edges, index_fill, temporal_degree, topological_overlap, get_adjacency_matrix


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
        self.nn_pi = nn.Linear(self.embedding_dim, self.categorical_dim, bias=False)

        self.rnn_nodes = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)
        self.rnn_comms = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)

        self.alpha_mean = nn.Embedding(self.num_samples, self.embedding_dim)
        self.alpha_std = nn.Embedding(self.num_samples, self.embedding_dim)

        self.subject_to_phi = nn.Linear(self.embedding_dim, self.num_nodes * self.embedding_dim)
        self.subject_to_beta = nn.Linear(self.embedding_dim, self.categorical_dim * self.embedding_dim)

        self.alpha_mean_prior = torch.zeros(self.embedding_dim)
        self.alpha_std_scalar = 1.

        self.decoder = nn.Sequential(nn.Linear(embedding_dim, num_nodes, bias=False))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, batch_data, train_prop=1, valid_prop=0.1, test_prop=0.1, temp=1.):
        loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0}
        for data in batch_data:
            subject_idx, dynamic_graph, _ = data
            subject_loss = self._forward(subject_idx, dynamic_graph, train_prop, valid_prop, test_prop, temp)
            for loss_name in loss.keys():
                loss[loss_name] += subject_loss[loss_name]
        return loss

    def _forward(self, subject_idx, batch_graphs, train_prop, valid_prop, test_prop, temp):

        loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0}
        edge_counter = 0

        time_len = len(batch_graphs)
        valid_time = math.floor(time_len * valid_prop)
        test_time = math.floor(time_len * test_prop)
        train_time = time_len - valid_time - test_time

        # train_snapshots = math.floor(train_time * train_prop)
        # train_start_idx = np.random.randint(0, train_time - train_snapshots + 1)

        # sample subject embedding from posterior
        alpha_mean_n = self.alpha_mean.weight[subject_idx]
        alpha_std_n = F.softplus(self.alpha_std.weight[subject_idx])
        # alpha_n = self._reparameterized_sample(alpha_mean_n, alpha_std_n)
        alpha_n = alpha_mean_n

        kld_alpha = self._kld_gauss(alpha_mean_n, alpha_std_n,
                                    self.alpha_mean_prior.to(self.device),
                                    self.alpha_std_scalar
                                    )

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.
        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)

        # Initialize the priors over nodes (phi) and communities (beta)
        phi_prior_mean = phi_0_mean
        beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        # iterate all over all edges in a graph at a single time point
        # for snapshot_idx in range(train_start_idx, train_start_idx + train_snapshots):
        for snapshot_idx in range(0, train_time):
            graph = batch_graphs[snapshot_idx]
            train_edges = [(u, v) for u, v in graph.edges()]
            if self.training:
                np.random.shuffle(train_edges)

            batch = torch.LongTensor(train_edges).to(self.device)
            assert batch.shape == (len(train_edges), 2)
            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))

            # Update node and community hidden states of GRUs
            nodes_in = torch.cat([phi_prior_mean, phi_prior_mean], dim=-1)

            nodes_in = nodes_in.view(1, self.num_nodes, 2 * self.embedding_dim)

            _, h_phi = self.rnn_nodes(nodes_in, h_phi)

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
            kld_z, BCE = self._vGraph_loss(recon, posterior_z, prior_z, c)
            kld_beta = self._kld_gauss(beta_mean_t, beta_std_t, beta_prior_mean, self.gamma)
            kld_phi = self._kld_gauss(phi_mean_t, phi_std_t, phi_prior_mean, self.sigma)

            loss['nll'] += BCE
            loss['kld_z'] += kld_z
            loss['kld_alpha'] += kld_alpha
            loss['kld_beta'] += kld_beta
            loss['kld_phi'] += kld_phi
            edge_counter += c.shape[0]

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample

        for loss_name in loss.keys():
            loss[loss_name] = loss[loss_name] / edge_counter

        return loss

    def predict_auc_roc_precision(self, subject_graphs, train_prop=1, valid_prop=0.1, test_prop=0.1, num_samples=1):
        aucroc = {'train': 0, 'valid': 0, 'test': 0}
        ap = {'train': 0, 'valid': 0, 'test': 0}
        nll = {'train': 0, 'valid': 0, 'test': 0}
        mse_to = 0
        mse_td = 0

        num_subjects = len(subject_graphs)
        for subject in subject_graphs:
            subject_idx, subject_graphs, gender_label = subject

            pred, label, _nll, _mse_to, _mse_td = self._predict_auc_roc_precision(subject_idx,
                                                                                  subject_graphs,
                                                                                  train_prop,
                                                                                  valid_prop,
                                                                                  test_prop,
                                                                                  num_samples)

            for status in ['train', 'valid', 'test']:
                if len(pred[status]) > 0:
                    aucroc[status] += roc_auc_score(label[status], pred[status]) / num_subjects
                    ap[status] += average_precision_score(label[status], pred[status]) / num_subjects
                    nll[status] += _nll[status].mean() / num_subjects
                    mse_to = _mse_to / num_subjects
                    mse_td = _mse_td / num_subjects

        return nll, aucroc, ap, mse_to, mse_td

    def _predict_auc_roc_precision(self, subject_idx, batch_graphs, train_prop, valid_prop, test_prop, num_samples):
        time_len = len(batch_graphs)
        valid_time = math.floor(time_len * valid_prop)
        test_time = math.floor(time_len * test_prop)
        train_time = time_len - valid_time - test_time

        # train_snapshots = math.floor(train_time * train_prop)
        # train_start_idx = train_time - train_snapshots

        pred = {'train': [], 'valid': [], 'test': []}
        label = {'train': [], 'valid': [], 'test': []}
        nll = {'train': [], 'valid': [], 'test': []}
        A = []
        pos_c_preds = []
        recons_c_pos = []
        ws_pos = []

        alpha_n = self.alpha_mean.weight[subject_idx].to(self.device)

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.

        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)

        # Initialize the priors over nodes (phi) and communities (beta)
        phi_prior_mean = phi_0_mean
        beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        # for i, graph in enumerate(batch_graphs[train_start_idx:]):
        for i, graph in enumerate(batch_graphs):
            # if i + train_start_idx < train_time:
            if i < train_time:
                status = 'train'
            # elif train_time <= i + train_start_idx < train_time + valid_time:
            elif train_time <= i < train_time + valid_time:
                status = 'valid'
            # elif train_time + valid_time <= i + train_start_idx < train_time + valid_time + test_time:
            else:
                status = 'test'
            # Sample edges
            pos_edges, neg_edges = sample_pos_neg_edges(graph, num_samples=num_samples)

            pos_batch = torch.LongTensor(pos_edges).to(self.device)
            neg_batch = torch.LongTensor(neg_edges).to(self.device)

            assert pos_batch.shape == (len(pos_edges), 2)
            assert neg_batch.shape == (len(neg_edges), 2)

            # Positive edge tensors
            w_pos = torch.cat((pos_batch[:, 0], pos_batch[:, 1]))
            c_pos = torch.cat((pos_batch[:, 1], pos_batch[:, 0]))

            # Negative edge tensors
            w_neg = torch.cat((neg_batch[:, 0], pos_batch[:, 1]))
            c_neg = torch.cat((neg_batch[:, 1], neg_batch[:, 0]))

            nodes_in = torch.cat([phi_prior_mean, phi_prior_mean], dim=-1)

            nodes_in = nodes_in.view(1, self.num_nodes, 2 * self.embedding_dim)

            # Update nodes (phi) hidden state
            _, h_phi = self.rnn_nodes(nodes_in, h_phi)

            comms_in = torch.cat([beta_prior_mean, beta_prior_mean], dim=-1)

            comms_in = comms_in.view(1, self.categorical_dim, 2 * self.embedding_dim)

            # Update communities (beta) hidden state
            _, h_beta = self.rnn_comms(comms_in, h_beta)

            # Sample node and community representations to be the means
            beta_sample = self.beta_mean(h_beta[-1])
            phi_sample = self.phi_mean(h_phi[-1])

            # Posterior distribution over the communities for each edge
            q_pos = F.softmax(self.nn_pi(phi_sample[w_pos] * phi_sample[c_pos]), dim=-1)
            # Weighted beta embedding based on the posterior community distribution
            beta_mixture_pos = torch.mm(q_pos, beta_sample)
            # Weighted beta embedding based on the posterior community distribution
            recon_c_pos = self.decoder(beta_mixture_pos)
            pos_c_pred = recon_c_pos.gather(1, c_pos.unsqueeze(dim=1)).squeeze(dim=-1).detach().cpu()

            q_neg = F.softmax(self.nn_pi(phi_sample[w_neg] * phi_sample[c_neg]), dim=-1)
            # Weighted beta embedding based on the posterior community distribution
            beta_mixture_neg = torch.mm(q_neg, beta_sample)
            recon_c_neg = self.decoder(beta_mixture_neg)
            neg_c_pred = recon_c_neg.gather(1, c_neg.unsqueeze(dim=1)).squeeze(dim=-1).detach().cpu()

            recon_c_softmax = F.log_softmax(recon_c_pos, dim=-1)
            bce = F.nll_loss(recon_c_softmax, c_pos, reduction='none')
            bce = bce.detach().cpu().numpy()

            pred[status] = np.hstack([pred[status], pos_c_pred.numpy(), neg_c_pred.numpy()])
            label[status] = np.hstack([label[status], np.ones(len(pos_c_pred)), np.zeros(len(neg_c_pred))])
            nll[status] = np.hstack([nll[status], bce])

            # If last validation step
            if train_time + valid_time - 1 <= i:
                A.append(torch.tensor(get_adjacency_matrix(graph), dtype=torch.float))
                pos_c_preds.append(pos_c_pred)
                recons_c_pos.append(recon_c_pos)
                ws_pos.append(w_pos)

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample

        A = torch.stack(A, dim=0).unsqueeze(0)
        pos_c_preds = torch.stack(pos_c_preds, dim=0).unsqueeze(0)
        recons_c_pos = torch.stack(recons_c_pos, dim=0).unsqueeze(0)
        ws_pos = torch.stack(ws_pos, dim=0).unsqueeze(0)

        A_pred = index_fill(torch.zeros_like(A), torch.ones_like(pos_c_preds), ws_pos, recons_c_pos.argmax(dim=-1))
        top_ov, top_ov_pred = topological_overlap(A), topological_overlap(A_pred)
        mse_to = F.mse_loss(top_ov_pred, top_ov, reduction="mean").item()
        temp_deg, temp_deg_pred = temporal_degree(A), temporal_degree(A_pred)
        mse_td = F.mse_loss(temp_deg_pred, temp_deg, reduction="mean").item()

        return pred, label, nll, mse_to, mse_td

    def predict_embeddings(self, subject_graphs, train_prop=1, valid_prop=0.1, test_prop=0.1):
        subjects = {}
        for subject in subject_graphs:
            subject_idx, subject_graphs, gender_label = subject

            subject_data = self._predict_embeddings(subject_idx, subject_graphs, train_prop, valid_prop, test_prop)
            subjects[subject_idx] = subject_data

        return subjects

    def _predict_embeddings(self, subject_idx, batch_graphs, train_prop, valid_prop, test_prop):
        time_len = len(batch_graphs)
        valid_time = math.floor(time_len * valid_prop)
        test_time = math.floor(time_len * test_prop)
        train_time = time_len - valid_time - test_time

        # train_snapshots = math.floor(train_time * train_prop)
        # train_start_idx = train_time - train_snapshots

        embeddings = {
            'alpha_embedding': None,
            'node_distribution_over_communities': {'train': [], 'valid': [], 'test': []},
            'beta_embeddings': {'train': [], 'valid': [], 'test': []},
            'phi_embeddings': {'train': [], 'valid': [], 'test': []}
        }

        alpha_n = self.alpha_mean.weight[subject_idx].to(self.device)
        alpha_embedding = alpha_n.cpu().detach().data.numpy()
        embeddings['alpha_embedding'] = alpha_embedding

        # inital values of phi and beta at time 0 per subject
        # TODO: We can just sample them from any distribution, e.g. N(0, I) as GRU takes subject embedding at each t now.

        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)

        # Initialize the priors over nodes (phi) and communities (beta)
        phi_prior_mean = phi_0_mean
        beta_prior_mean = beta_0_mean

        # GRU hidden states for node and community embeddings
        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        # for i, graph in enumerate(batch_graphs[train_start_idx:]):
        for i, graph in enumerate(batch_graphs):
            # if i + train_start_idx < train_time:
            if i < train_time:
                status = 'train'
            # elif train_time <= i + train_start_idx < train_time + valid_time:
            elif train_time <= i < train_time + valid_time:
                status = 'valid'
            # elif train_time + valid_time <= i + train_start_idx < train_time + valid_time + test_time:
            else:
                status = 'test'

            nodes_in = torch.cat([phi_prior_mean, phi_prior_mean], dim=-1)

            nodes_in = nodes_in.view(1, self.num_nodes, 2 * self.embedding_dim)

            _, h_phi = self.rnn_nodes(nodes_in, h_phi)

            comms_in = torch.cat([beta_prior_mean, beta_prior_mean], dim=-1)

            comms_in = comms_in.view(1, self.categorical_dim, 2 * self.embedding_dim)

            _, h_beta = self.rnn_comms(comms_in, h_beta)

            # Sample node and community representations
            beta_sample = self.beta_mean(h_beta[-1])
            phi_sample = self.phi_mean(h_phi[-1])

            node_distrib_over_communities_t = self.decoder(beta_sample)
            node_distrib_over_communities_t = F.softmax(
                node_distrib_over_communities_t,
                dim=-1).cpu().detach().data.numpy()

            embeddings['node_distribution_over_communities'][status].append(
                node_distrib_over_communities_t)
            embeddings['beta_embeddings'][status].append(beta_sample.cpu().detach().data.numpy())
            embeddings['phi_embeddings'][status].append(phi_sample.cpu().detach().data.numpy())

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample

        return embeddings

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
