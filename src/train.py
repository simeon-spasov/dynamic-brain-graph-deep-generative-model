import logging
from pathlib import Path

import numpy as np
import torch

from .dataset import data_loader


def train(model, dataset,
          save_path=Path.cwd() / "models",
          learning_rate=1e-4,
          temp=1.,
          temp_min=0.05,
          num_epochs=1001,
          anneal_rate=0.0003,
          batch_size=1,
          weight_decay=0.,
          train_prop=1.,
          valid_prop=0.1,
          test_prop=0.1,
          device=torch.device("cpu")
          ):
    """
    Train the given model using the provided dataset and parameters.

    :param model: The model to be trained.
    :param dataset: The dataset for training.
    :param save_path: The path to save the model and results.
    :param learning_rate: Learning rate for the optimizer.
    :param temp: Initial temperature for Gumbel-softmax.
    :param temp_min: Minimum temperature for Gumbel-softmax.
    :param num_epochs: Number of epochs for training.
    :param anneal_rate: Annealing rate for temperature.
    :param batch_size: Batch size for training.
    :param weight_decay: Weight decay for the optimizer.
    :param train_prop: Proportion of data used for training.
    :param valid_prop: Proportion of data used for validation.
    :param test_prop: Proportion of data used for testing.
    :param device: Device to use for training (e.g., 'cpu' or 'cuda').
    """
    # Create save path if it doesn't exist
    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.warning(f"Folder to save model weights already exists at {save_path}. "
                        f"Existing model parameters with same name "
                        f"in the directory will be overriden.")
    else:
        logging.info(f"Folder to save model weights was created at {save_path}")

    logging.info(f"Model save path is: {save_path}. ")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    # Move model to device
    model.to(device)

    # Initialize variables to track best losses
    best_nll = float('inf')
    best_nll_train = float('inf')

    # Begin training
    for epoch in range(num_epochs):
        logging.debug(f"Starting epoch {epoch}")
        np.random.shuffle(dataset)
        model.train()

        running_loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0}

        for batch_graphs in data_loader(dataset, batch_size):
            optimizer.zero_grad()

            batch_loss = model(batch_graphs,
                               train_prop=train_prop,
                               valid_prop=valid_prop,
                               test_prop=test_prop,
                               temp=temp)

            loss = (batch_loss['nll'] + batch_loss['kld_z']) / len(batch_graphs)
            loss.backward()
            optimizer.step()
            for loss_name in running_loss.keys():
                running_loss[loss_name] += batch_loss[loss_name].cpu().detach().data.numpy() / len(dataset)

        logging.info(f"Epoch {epoch} | {running_loss}")

        model.eval()

        nll, aucroc, ap = model.predict_auc_roc_precision(
            batch_graphs,
            train_prop=train_prop,
            valid_prop=valid_prop,
            test_prop=test_prop)

        logging.info(f"Epoch {epoch} | "
                     f"train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']}| "
                     f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
                     f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']} | ")
        if nll['valid'] < best_nll:
            embeddings = model.predict_embeddings(dataset,
                                                  train_prop=train_prop,
                                                  valid_prop=valid_prop,
                                                  test_prop=test_prop)
            logging.info(f"Saving model.")
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                Path(save_path) / "checkpoint.pt"
            )

            np.save(Path(save_path) / "results.npy",
                    {
                        'nll': nll,
                        'aucroc': aucroc,
                        'ap': ap,
                        'embeddings': embeddings
                    })

            best_nll = nll['valid']

        if nll['train'] < best_nll_train:
            embeddings = model.predict_embeddings(dataset,
                                                  train_prop=train_prop,
                                                  valid_prop=valid_prop,
                                                  test_prop=test_prop)
            logging.info(f"Saving model (best train).")
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                Path(save_path) / "checkpoint_best_train.pt"
            )

            np.save(Path(save_path) / "results_best_train.npy",
                    {
                        'nll': nll,
                        'aucroc': aucroc,
                        'ap': ap,
                        'embeddings': embeddings
                    })

            best_nll_train = nll['train']

        if epoch % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * epoch), temp_min)
            logging.debug(f"Updating temperature for Gumbel-softmax to {temp}")
            learning_rate *= .99
            logging.debug(f"Updating learning rate to {learning_rate}")

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

    logging.info(f"Finished training.")
