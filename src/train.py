import time
from pathlib import Path

import dill
import numpy as np
import torch
import torch.optim as opt
from omegaconf import OmegaConf

from .log import Logger


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
    # mean over time, edges and batch
    nll = nll_pos.mean(dim=(-3, -2, -1))
    
    # sum over latent dimension and then mean over batch
    kl_alpha = kl_alpha.sum(dim=-1).mean(dim=-1)
    
    # sum over latent dimension and then mean over time and batch
    kl_beta = kl_beta.sum(dim=(-2)).mean(dim=(-1, -2))
    
    # sum over latent dimension and time: B x N x Z x T -> B x N 
    kl_phi = kl_phi.sum(dim=(-2, -1))
    # mean over number of nodes and batch
    kl_phi = kl_phi.mean(dim=(-1, -2))

    # sum over latent dimension and time: B x E x K x T -> B x E 
    kl_z = kl_z.sum(dim=(-2, -1))
    # mean over number of edges and batch
    kl_z = kl_z.mean(dim=(-2, -1))

    elbo = (nll_pos + (self.neg_scale * nll_neg)) / (1 + self.neg_scale)
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

def step(model, dataloader, optimiser=None, device=None, args=None):
    model.to(device)
    model.eval() if optimizer is None else model.train()
 
    metrics = dict(elbo=0., nll=0., kl_alpha=0., kl_beta=0., kl_phi=0., kl_z=0.)\

    for batch in dataloader:
        x = batch["x"].to(device)
        idx = batch["..."].to(device)
  
        _nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z = model(idx, x)
        elbo, elbo_parts = loss_fnc(_nll, _kl_alpha, _kl_beta, _kl_phi, _kl_z, 
                                    args.train.gamma_alpha, args.trarin.gamma_beta, 
                                    args.train.gamma_phi., args.train.gamma_z, model.warmup)
        
        # compute gradients and update model parameters
        if optimiser is not None:
            optimiser.zero_grad()
            elbo.backward()
            optimiser.step()
        
        # accumulate loss
        metrics["elbo"] += elbo.item() / len(dataloader) 
        for key, value in elbo_parts.items():
            metrics[key] += value / len(dataloader) 

    return metrics

def train(model, train_data, valid_data, optimizer, scheduler, device, args)
    # make directories
    results_dir = Path("./results") / args.exp_id 
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("./models") / args.exp_id 
    models_dir.mkdir(parents=True, exist_ok=True)
    # save config
    OmegaConf.save(args, models_dir / "config.yaml")

    # logging
    train_logger = Logger("train", results_dir)
    valid_logger = Logger("valid", results_dir)

    # dataloaders
    trainloader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True, num_workers=args.data.num_cpus,
                             pin_memory=True if device.type == "cuda" else False) 
    validloader = DataLoader(valid_data, batch_size=args.train.batch_size, shuffle=False, num_workers=args.data.num_cpus,
                             pin_memory=True if device.type == "cuda" else False) 
    
    # early stopping init
    counter = 0
    best_valid_nll = torch.inf
  
    for epoch in range(args.train.num_epochs):
        start_time = time.time() 
        # check warmup epoch
        model.warmup = True if epoch < args.train.warmup else False
        # check valid epoch
        valid = True if (epoch % args.train.valid_every == 0) else False
        
        # train epoch
        train_metrics = step(model, train_loader, optimizer, device, args)
        train_logger.update(train_logger)

        # valid epoch
        if not model.warmup and valid:
            with torch.no_grad():
                valid_metrics = step(model, valid_loader, None, device, args)
            valid_logger.update(valid_metrics)
            
            # early stopping
            valid_nll = valid_metrics["nll"]
            if (best_valid_nll - valid_nll) > args.train.delta:
                counter = 0
                best_valid_nll = valid_nll
                torch.save(model, models_dir / "model.pth.tar", pickle_module=dill)
            else:
                counter += 1
                if (counter == args.train.patience):
                    break 

        # temperature annealing
        if epoch >= args.train.anneal_after:
            model.temp = np.maximum(model.temp * np.exp(-args.train.anneal_rate * epoch), args.train.min_temp)

        # update learning rate
        if scheduler is not None:
            scheduler.step()       
        
        # print output
        if args.train.verbose:
            end_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            to_print = "{} | epoch {:3d} of {:3d} | elbo {:06.2f} | nll {:06.2f} | kl alpha {:06.2f} | "
                       "kl beta {:06.2f} | kl phi {:06.2f} | kl z {:06.2f} | time {} "
            if model.warmup: to_print = to_print + "#" 
            if counter == 0 and not model.warmup: to_print = to_print + "*" 
            print(to_print.format(args.exp_id, epoch + 1, num_epochs, *[train_metrics[idx] for idx in range(len(train_metrics.metrics))], end_time))

