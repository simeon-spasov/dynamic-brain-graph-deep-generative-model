import time
import wandb
import torch

import numpy as np
from torch.optim import Adam


def _save_model(model, save_dir, is_checkpoint=False):
    if is_checkpoint:
        filepath = save_dir / "best_model.pth.tar"
    else:
        filepath = save_dir / "last_model.pth.tar"
    
    torch.save({"state_dict": model.state_dict()}, filepath)


class Accumulator:
    metrics = ["elbo", "nll_pos", "nll_neg", "kl_alpha", "kl_beta", "kl_phi", "kl_z"]

    def __init__(self):
        self.data = [0.0] * len(self.metrics)

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def logging(self, epoch):
        log_dict = {key: value for key, value in zip(self.metrics, self.data)} 
        log_dict["epoch"] = epoch

        return log_dict

    def __getitem__(self, idx):
        return self.data[idx]


def _train_epoch(model, dataloader, optimiser, device=torch.device("cpu"), valid=False):
    model.train()
    model.freeze(valid)

    metrics = Accumulator()

    for subject_idx, pos_edges, pos_edge_weights, neg_edges in dataloader:

        subject_idx = subject_idx.to(device)
        pos_edges = pos_edges.to(device)
        pos_edge_weights = pos_edge_weights.to(device)
        neg_edges = neg_edges.to(device)
    
        optimiser.zero_grad()

        loss, loss_parts = model(subject_idx, pos_edges, pos_edge_weights, neg_edges)

        loss.backward()

        optimiser.step()
        
        metrics.add(loss / len(dataloader), *(loss_parts / len(dataloader)))
        
    return metrics


def train(model, train_loader, valid_loader, num_epochs, learning_rate, save_dir, anneal_after=100, min_tau=0.1, anneal_rate=1e-5,
          warmup=10, valid_every=10, device=torch.device("cpu"), checkpoint_best=True, verbose=True, wb_logging=False):

    model.to(device)

    optimiser = Adam(params=model.parameters(), lr=learning_rate)

    if wb_logging: wandb.watch(model)
        
    best_valid_elbo = float("inf")
    is_best = False

    for epoch in range(num_epochs):
        start_time = time.time() 
        
        model.warmup = True if epoch < warmup else False
        if (epoch % valid_every == 0): valid = True 

        train_metrics = _train_epoch(model, train_loader, optimiser, device)
        
        if not model.warmup and valid:

            valid_metrics = _train_epoch(model, valid_loader, optimiser, device, valid)
            
            valid_elbo = valid_metrics[0]
            is_best = valid_elbo < best_valid_elbo
    
            if is_best and checkpoint_best:
                
                _save_model(model, save_dir, is_best)
                best_valid_elbo = valid_elbo
                
        if epoch >= anneal_after:
            model.tau = np.maximum(model.tau * np.exp(-anneal_rate * epoch), min_tau)
            
        if verbose:
            
            end_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            to_print = "{} | epoch {:4d} of {:4d} | elbo {:06.2f} | nll_pos {:06.2f} | nll_neg {:06.2f} | kl_alpha {:06.2f} | kl_beta {:06.2f} | kl_phi {:06.2f} | kl_z {:06.2f} | time: {} "
            if model.warmup: to_print = to_print + "| warm-up" 
            if is_best and checkpoint_best and not model.warmup: to_print = to_print + "| *" 
            print(to_print.format(save_dir.stem, epoch + 1, num_epochs, *[train_metrics[idx] for idx in range(len(train_metrics.metrics))], end_time))

        if wb_logging:

            train_log = train_metrics.logging(epoch)
            train_log["tau"] = model.tau
            log_dict = dict(train = train_log)
            
            if not model.warmup and valid:
                valid_log = valid_metrics.logging(epoch)
                log_dict["valid"] = valid_log
            
            wandb.log(log_dict)
 
    _save_model(model, save_dir)

    return model