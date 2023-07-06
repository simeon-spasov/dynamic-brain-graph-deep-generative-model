import argparse
import logging
from pathlib import Path

import torch

from src.dataset import load_dataset
from src.inference import inference
from src.model import Model
from src.train import train


def main(args):
    logging.basicConfig(
        filename=f'fMRI_{args.dataset}_{args.trial}.log',
        format='%(levelname)s:%(message)s',
        filemode='w',
        level=logging.DEBUG
    )

    # Setup
    data_dir = "./data"
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        if args.gpu:
            logging.warning(f'Torch cannot find GPU. Using device: {device} instead.')

    logging.info(f'Using device: {device}')

    # Hyperparameters
    dataset_args = dict(
        dataset=args.dataset,
        window_size=30,
        window_stride=30,
        measure="correlation",
        top_percent=5
    )

    model_args = dict(
        sigma=1.0,
        gamma=0.1,
        categorical_dim=args.categorical_dim,
        embedding_dim=128
    )

    # Batch size = 1 converges fastest
    train_args = dict(
        num_epochs=1001,
        save_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        batch_size=1,
        learning_rate=1e-4,
        device=device,
        temp_min=0.05,
        anneal_rate=3e-4,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        temp=1.0
    )

    inference_args = dict(
        load_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        save_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        device=device,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        num_samples=1
    )

    # Log all dataset parameters.
    logging.debug('Dataset args: %s', dataset_args)
    # Log all model parameters.
    logging.debug('Model args: %s', model_args)
    # Log all training setup parameters.
    logging.debug('Train args: %s', train_args)
    # Log all inference setup parameters.
    logging.debug('Inference args: %s', inference_args)

    # Dataset
    logging.info('Loading data.')
    dataset = load_dataset(**dataset_args, data_dir=data_dir)
    num_subjects, num_nodes = len(dataset), dataset[0][1][0].number_of_nodes()
    logging.info(f'{num_subjects} subjects with {num_nodes} nodes each.')

    # model
    model = Model(num_subjects, num_nodes, **model_args, device=device)

    if args.command != 'inference':
        # Train
        logging.info('Starting training.')
        train(model, dataset, **train_args)
        logging.info('Finished training.')

    else:
        logging.info('Starting inference.')
        inference(model, dataset, **inference_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBGDGM model.')
    parser.add_argument('--dataset', required=True, type=str, choices=['ukb', 'hcp'])
    parser.add_argument('--categorical-dim', required=True, type=int)
    parser.add_argument('--valid-prop', default=0.1, type=float)
    parser.add_argument('--test-prop', default=0.1, type=float)
    parser.add_argument('--trial', required=True, type=int)
    parser.add_argument('--gpu', default=None, type=int, choices=[0, 1])

    subparsers = parser.add_subparsers(dest='command')
    parser_foo = subparsers.add_parser('inference')

    args = parser.parse_args()
    main(args)
