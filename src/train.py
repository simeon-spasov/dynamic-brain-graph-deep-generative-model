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
          valid_prop=0.1,
          test_prop=0.1,
          device=torch.device("cpu")
          ):
    """
    Trains the provided model using the given dataset and parameters.

    Args:
        model (nn.Module): The model instance to be trained.
        dataset (list): The dataset used for training the model.
        save_path (Path, optional): The directory where the model and the results will be saved. Default is the current working directory under the 'models' subdirectory.
        learning_rate (float, optional): The learning rate for the optimizer. Default is 1e-4.
        temp (float, optional): The initial temperature for Gumbel-softmax. Default is 1.0.
        temp_min (float, optional): The minimum temperature for Gumbel-softmax. Default is 0.05.
        num_epochs (int, optional): The number of epochs for training. Default is 1001.
        anneal_rate (float, optional): The annealing rate for the temperature. Default is 0.0003.
        batch_size (int, optional): The batch size for training. Default is 1.
        weight_decay (float, optional): The weight decay for the optimizer. Default is 0.0.
        valid_prop (float, optional): The proportion of data to be used for validation. Default is 0.1.
        test_prop (float, optional): The proportion of data to be used for testing. Default is 0.1.
        device (torch.device, optional): The device to use for training (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns: None. The function saves the trained model and results to the specified save_path. The model parameters
        are saved every time the validation negative log-likelihood improves. The final model parameters,
        optimizer parameters, negative log-likelihoods, AUC-ROC, AP, and embeddings are saved for the best model
        according to validation negative log-likelihood and the best model according to training negative log-likelihood.
        The model is saved as 'checkpoint.pt' and the results are saved as 'results.npy'. The model and results for the
        best model according to training negative log-likelihood are saved as 'checkpoint_best_train.pt' and
        'results_best_train.npy', respectively.
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
                               valid_prop=valid_prop,
                               test_prop=test_prop,
                               temp=temp)

            loss = (batch_loss['nll'] + batch_loss['kld_z'] + 0*batch_loss['kld_alpha'] + 0*batch_loss['kld_beta'] + 0*batch_loss['kld_phi']) / len(batch_graphs)
            loss.backward()
            optimizer.step()
            for loss_name in running_loss.keys():
                running_loss[loss_name] += batch_loss[loss_name].cpu().detach().data.numpy() / len(dataset)

        logging.info(f"Epoch {epoch} | {running_loss}")

        model.eval()

        nll, aucroc, ap = model.predict_auc_roc_precision(
            batch_graphs,
            valid_prop=valid_prop,
            test_prop=test_prop)

        logging.info(f"Epoch {epoch} | "
                     f"train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']}| "
                     f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
                     f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']} | ")
        if nll['valid'] < best_nll:
            embeddings = model.predict_embeddings(dataset,
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
