from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import save_results


def test(model, test_data,  device, args):
    # make directories
    results_dir = Path("./results") / args.exp_id 

    # dataloaders
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True if device.type == "cuda" else False) 

    model.to(device)
    model.eval()
    model.inference = True

    beta, phi, comm_nodes = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            idx = batch["idx"].to(device)
            # (1, num_edges, num_windows)
            nll, _, _, _, _ = model(x, idx)
            # save average negative log likelihood
            save_results({"nll": nll.mean().item()}, "test", results_dir)
            # community embeddings
            # (batch_size, num_communities, beta_dim, num_windows)
            _beta = torch.stack([q.mean for q in model.q_beta], dim=-1)
            # community node distribution
            # (batch_size, num_communities, beta_dim, num_windows) -> (batch_size, num_communities, num_nodes, num_windows)
            _comm_nodes = model.fc_p_c(_beta.transpose(-2, -1)).transpose(-2, -1)
            # node embeddings
            # (batch_size, num_nodes, phi_dim, num_windows)
            _phi = torch.stack([q.mean for q in model.q_phi], dim=-1)
            
            beta += [_beta.detach().cpu()]
            phi += [_phi.detach().cpu()]
            comm_nodes += [_comm_nodes.detach().cpu()]

    alpha = model._parameterize_posterior_alpha(None).mean.detach().cpu().numpy()
    beta = torch.concat(beta, dim=0).numpy()
    phi = torch.concat(phi, dim=0).numpy()
    comm_nodes = torch.concat(comm_nodes, dim=0).numpy()
    
    np.savez(results_dir / "test_results", alpha=alpha, beta=beta, phi=phi, comm_nodes=comm_nodes)
    

