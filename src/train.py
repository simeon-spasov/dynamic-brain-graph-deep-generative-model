import numpy as np
import torch

from .dataset import data_loader


def train(model, dataset, num_epochs=1000, batch_size=25, ..., device=torch.device("cpu")):
    optimizer = 

    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        np.random.shuffle(dataset)

        running_loss = 0
        for batch_graphs in data_loader(dataset, batch_size):
            optimizer.zero_grad()
            loss = model(batch_graphs)
            loss.backward()
            optimizer.step()
            running_loss += loss / ....
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + '/fmri_mod.pt')

        if epoch % 100 == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
            cur_lr *= .99

            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lr

    torch.save(model.state_dict(), save_path + '/fmri_mod.pt')
