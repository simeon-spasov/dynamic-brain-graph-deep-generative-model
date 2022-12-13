import logging

import torch

from src.dataset import load_dataset
from src.inference import inference
from src.model import Model
from src.train import train


def main():
    logging.basicConfig(
        filename='fMRI.log',
        format='%(levelname)s:%(message)s',
        filemode='w',
        level=logging.DEBUG)

    # setup
    data_dir = "./data"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 12345

    # hyperparameters
    dataset_args = dict(dataset="hcp",
                        window_size=30,
                        window_stride=30,
                        measure="correlation",
                        top_percent=5)
    model_args = dict(sigma=1.,
                      gamma=0.1,
                      categorical_dim=3,
                      embedding_dim=128)
    # Need batch size = 1 to optimize per subject.
    train_args = dict(num_epochs=2001,
                      batch_size=1,
                      learning_rate=1e-3,
                      device=device,
                      temp_min=0.1,
                      anneal_rate=3e-5,
                      train_prop=0.3,
                      valid_prop=0.1,
                      test_prop=0.1,
                      temp=1.)

    # Log all dataset parameters.
    logging.debug('Dataset args: %s', dataset_args)
    # Log all model parameters.
    logging.debug('Model args: %s', model_args)
    # Log all training setup parameters.
    logging.debug('Train args: %s', train_args)

    # dataset
    logging.info('Loading data.')
    dataset = load_dataset(**dataset_args, data_dir=data_dir)
    experiment_dataset = dataset
    num_subjects, num_nodes = len(experiment_dataset), experiment_dataset[0][1][0].number_of_nodes()
    logging.info(f'{num_subjects} subjects with {num_nodes} nodes each.')

    # model
    model = Model(num_subjects, num_nodes, **model_args, device=device)

    # train
    logging.info('Starting training.')
    train(model, experiment_dataset, **train_args)

    logging.info('Running inference...')
    inference(model, experiment_dataset, device=device)

    logging.info('Finished script.')


if __name__ == '__main__':
    main()
