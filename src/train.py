import logging
from pathlib import Path

import numpy as np
import torch

from .dataset import data_loader


def train(model, dataset,
          save_path=Path.cwd() / "train_model",
          learning_rate=0.005,
          temp=1.,
          temp_min=0.1,
          num_epochs=1000,
          anneal_rate=0.00003,
          batch_size=25,
          weight_decay=0.,
          device=torch.device("cpu")
          ):
    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.warning(f"Folder to save model weights already exists at {save_path}. "
                        f"Existing model parameters with same name "
                        f"in the directory will be overriden.")
    else:
        logging.info(f"Folder to save model weights was created at {save_path}")

    model_save_path = Path(save_path) / "fmri_model.pt"
    logging.info(f"Model save path is: {save_path}. ")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        logging.debug(f"Starting epoch {epoch}")
        np.random.shuffle(dataset)

        running_loss = 0
        for batch_graphs in data_loader(dataset, batch_size):
            optimizer.zero_grad()
            loss = model(batch_graphs, temp=temp)
            loss.backward()
            optimizer.step()
            running_loss += loss / len(batch_graphs)

        logging.debug(f"Epoch {epoch}: Running loss is {running_loss}\n")

        if epoch % 10 == 0:
            logging.info(f"Saving model.")
            torch.save(model.state_dict(), str(model_save_path))

        if epoch % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * epoch), temp_min)
            logging.debug(f"Updating temperature for Gumbel-softmax to {temp}")
            learning_rate *= .99
            logging.debug(f"Updating learning rate to {learning_rate}")

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

    logging.info(f"Saving model.")
    torch.save(model.state_dict(), str(model_save_path))
