import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.utils import edge_sampler
from sklearn.model_selection import train_test_split


def get_synthetic(data_dir, valid_prop, test_prop, num_pos_edges, num_neg_edges, seed, **kwargs):
    
    filepaths = list((data_dir / "synthetic").rglob("*.npz"))

    train_filepaths, test_filepaths = train_test_split(filepaths, test_size=test_prop, shuffle=True, random_state=seed)
    train_filepaths, valid_filepaths = train_test_split(train_filepaths, test_size=valid_prop, shuffle=True, random_state=seed)

    train = SyntheticDataset(train_filepaths, num_pos_edges, num_neg_edges, **kwargs)
    valid = SyntheticDataset(valid_filepaths, num_pos_edges, num_neg_edges, **kwargs)
    test = SyntheticDataset(test_filepaths, num_pos_edges, num_neg_edges, **kwargs)

    return {"train": train, "valid": valid, "test": test}
    

class SyntheticDataset(Dataset):

    def __init__(self, filepaths, num_pos_edges=100, num_neg_edges=100):
        self.filepaths = filepaths
        self.num_pos_edges = num_pos_edges
        self.num_neg_edges = num_neg_edges
        self._get_data_dims()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        graph_timeseries = self._load(self.filepaths[idx])
        
        pos_edges, pos_edge_weights, neg_edges = self._sample_edges(graph_timeseries)
        subject_idx = torch.LongTensor([idx])

        return subject_idx, pos_edges, pos_edge_weights, neg_edges

    def _load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        return [data[k].tolist()[0] for k in list(data.keys())]

    def _get_data_dims(self):
        graph_timeseries = self._load(self.filepaths[0])
        self.num_samples = len(self.filepaths)
        self.num_timesteps = len(graph_timeseries)
        self.num_nodes = graph_timeseries[0].shape[0]

    def _sample_edges(self, graph_timeseries):

        pos_edges = []
        pos_edge_weights = []
        neg_edges = []

        for graph in graph_timeseries:
            p_edges, p_edge_weights, n_edges = edge_sampler(graph, self.num_pos_edges, self.num_neg_edges)
            pos_edges += [p_edges]
            pos_edge_weights += [p_edge_weights]
            neg_edges += [n_edges]

        pos_edges = torch.tensor(pos_edges).long()
        pos_edge_weights = torch.tensor(pos_edge_weights).float()
        neg_edges = torch.tensor(neg_edges).long()

        return pos_edges, pos_edge_weights, neg_edges