import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.optim as opt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .utils import save_results


def get_optimizer(name):
    op = {"sgd": opt.SGD, "adam": opt.Adam, "adamw": opt.AdamW}
    if name in op.keys():
        return op[name]
    else:
        raise NotImplementedError("unknown optimizer '{}'. Only {} are supported".format(name, ", ".join(list(op.keys()))))

def get_scheduler(name):
    sc = {"step": opt.lr_scheduler.StepLR, "multi_step": opt.lr_scheduler.MultiStepLR}
    if name in sc.keys():
        return sc[name]
    else:
        raise NotImplementedError("unknown scheduler '{}'. Only {} are supported".format(name, ", ".join(list(sc.keys()))))

def loss_fnc(nll, kl_alpha, kl_beta, kl_phi, kl_z, gamma_alpha=1., gamma_beta=1., 
             gamma_phi=1., gamma_z=1., warmup=False): 
    # mean over num_windows, num_edges and batch_size
    # (batch_size, num_edges, num_windows) 
    nll = nll.mean(dim=(0, 1, 2))
    # sum over alpha_dim and mean over batch_size
    # (batch_size, alpha_dim) 
    kl_alpha = kl_alpha.sum(dim=1).mean(dim=0) 
    # sum over beta_dim and mean over num_windows, num_communities, and batch_size 
    # (batch_size, num_communities, beta_dim, num_windows) 
    kl_beta = kl_beta.sum(dim=2).mean(dim=(0, 1, 2))
    # sum over phi_dim and mean over num_windows, num_nodes, and batch_size
    # (batch_size, num_nodes, phi_dim, num_windows) -> 
    kl_phi = kl_phi.sum(dim=2).mean(dim=(0, 1, 2))
    # mean over num_windows, num_edges and batch_size
    # (batch_size, num_edges, num_windows) 
    kl_z = kl_z.mean(dim=(0, 1, 2))

    elbo = nll
    if not warmup:
        elbo += gamma_alpha * kl_alpha
        elbo += gamma_beta * kl_beta
        elbo += gamma_phi * kl_phi
        elbo += gamma_z * kl_z
        elbo_parts = dict(nll=nll.item(), kl_alpha=kl_alpha.item(), kl_beta=kl_beta.item(), 
                          kl_phi=kl_phi.item(), kl_z=kl_z.item())
    else:
        elbo_parts =  dict(nll=nll.item(), kl_alpha=0, kl_beta=0, kl_phi=0, kl_z=0)
            
    return elbo, elbo_parts

def train_step(model, dataloader, optimizer=None, warmup=False, device=None, args=None):
    model.to(device)
    model.train()
 
    metrics = OrderedDict(elbo=0., nll=0., kl_alpha=0., kl_beta=0., kl_phi=0., kl_z=0.)

    for batch in dataloader:
        x = batch["x"].to(device)
        idx = batch["idx"].to(device)
    
        _nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z = model(x, idx)
        elbo, elbo_parts = loss_fnc(_nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z, 
                                    args.train.gamma_alpha, args.train.gamma_beta, 
                                    args.train.gamma_phi, args.train.gamma_z, warmup)
        
        # compute gradients and update model parameters
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
        
        # accumulate loss
        metrics["elbo"] += elbo.item() / len(dataloader) 
        for key, value in elbo_parts.items():
            metrics[key] += value / len(dataloader)
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["temp"] = model.temp

    return metrics

def valid_step(model, dataloader, device=None):
    model.to(device)
    model.eval()

    metrics = OrderedDict(nll=0.)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            idx = batch["idx"].to(device)
  
            _nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z = model(x, idx)
            elbo, elbo_parts = loss_fnc(_nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z)

            metrics["nll"] += elbo_parts["nll"] / len(dataloader)
    
    return metrics

def train(model, train_data, valid_data, optimizer, scheduler, device, args):
    # make directories
    results_dir = Path("./results") / args.exp_id 
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("./models") / args.exp_id 
    models_dir.mkdir(parents=True, exist_ok=True)
    # save config
    OmegaConf.save(args, models_dir / "config.yaml")

    # dataloaders
    train_loader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True, num_workers=args.data.num_cpus,
                              pin_memory=True if device.type == "cuda" else False) 
    valid_loader = DataLoader(valid_data, batch_size=args.train.batch_size, shuffle=False, num_workers=args.data.num_cpus,
                              pin_memory=True if device.type == "cuda" else False) 
    
    warmup = False
    valid = False

    # early stopping init
    counter = 0
    best_valid_nll = torch.inf
  
    for epoch in range(args.train.num_epochs):
        start_time = time.time() 
        # check warmup epoch
        warmup = True if epoch < args.train.warmup else False
        # check valid epoch
        valid = True if (epoch % args.train.valid_every == 0) else False
        
        # train epoch
        train_metrics = train_step(model, train_loader, optimizer, warmup, device, args)
        save_results(train_metrics, "train", results_dir)

        # valid epoch
        if not warmup and valid:
            valid_metrics = valid_step(model, valid_loader, device)
            save_results(valid_metrics, "valid", results_dir)
            
            # early stopping
            valid_nll = valid_metrics["nll"]
            if (best_valid_nll - valid_nll) > args.train.delta:
                counter = 0
                best_valid_nll = valid_nll
                torch.save(model, models_dir / "best_model.pth.tar")
            else:
                counter += 1
                if (counter == args.train.patience):
                    break 
        
        # temperature annealing
        if epoch - args.train.warmup >= args.train.anneal_after:
            model.temp = np.maximum(model.temp * np.exp(-args.train.anneal_rate * epoch), args.train.min_temp)

        # update learning rate
        if scheduler is not None:
            scheduler.step()       
        
        # print output
        if args.train.verbose:
            end_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
            to_print = "{} | {:3d} of {:3d} | elbo {:05.2f} | nll {:05.2f} | kl alpha {:05.2f} | "\
                       "kl beta {:05.2f} | kl phi {:05.2f} | kl z {:05.2f} | {} | "
            if warmup: to_print = to_print + "#" 
            if counter == 0 and not warmup: to_print = to_print + "*" 
            print(to_print.format(args.exp_id, epoch + 1, args.train.num_epochs, *train_metrics.values(), end_time))

    torch.save(model, models_dir / "model.pth.tar")
