import logging

import torch

from src.dataset import load_dataset
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
                        window_size=90,
                        window_stride=40,
                        measure="correlation",
                        top_percent=5)
    model_args = dict(sigma=1.,
                      gamma=0.1,
                      categorical_dim=3,
                      embedding_dim=32)
    train_args = dict(num_epochs=2001,
                      batch_size=1,
                      learning_rate=1e-3,
                      device=device,
                      temp_min=0.1,
                      anneal_rate=3e-5,
                      temp=1.)

    # dataset
    logging.info('Loading data.')
    dataset = load_dataset(**dataset_args, data_dir=data_dir)
    experiment_dataset = dataset[:10]
    num_subjects, num_nodes = len(experiment_dataset), experiment_dataset[0][1][0].number_of_nodes()
    logging.info('{num_subjects} subjects with {num_nodes} nodes each.')

    # model
    model = Model(num_subjects, num_nodes, **model_args, device=device)

    # train
    logging.info('Starting training.')
    train(model, experiment_dataset, **train_args)
    logging.info('Finished script.')


if __name__ == '__main__':
    main()
