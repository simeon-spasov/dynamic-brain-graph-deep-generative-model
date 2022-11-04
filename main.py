import pickle
import random

import numpy as np
import torch

from src.dataset import load_dataset
from src.model import Model
from src.train import train

# setup
data_dir = "../data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 12345

# hyperparameters
dataset_args = dict(dataset="hcp", 
                    window_size=30, 
                    window_stride=10, 
                    measure="correlation", 
                    top_percent=5)
model_args = dict(sigma=1., 
                  gamma=0.1, 
                  categorical_dim=3, 
                  embedding_dim=32)
train_args = dict(num_epochs=10001, 
                  batch_size=25, 
                  learning_rate=5e-3, 
                  lamda=100, 
                  temp_min=0.1, 
                  anneal_rate=3e-5,
                  temp=1.)

# dataset
dataset = load_dataset(**dataset_args, data_dir=data_dir)
num_subjects, num_nodes = len(dataset), dataset[0][1][0].number_of_nodes()

# model
model = Model(num_subjects, num_nodes, **model_args, device=device)

# train
model = train(model, dataset, **train_args)
