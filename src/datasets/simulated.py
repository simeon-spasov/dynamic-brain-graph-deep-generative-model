import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.utils import edge_sampler
from sklearn.model_selection import train_test_split


def get_simulated(data_dir, valid_prop, test_prop, num_pos_edges, num_neg_edges, seed):
    
    filepaths = list(data_dir.rglob("*.npz"))
    labels = [int(f.stem.split("-")[-1]) for f in filepaths]

    train_filepaths, test_filepaths = train_test_split(filepaths, stratify=labels, test_size=test_prop, shuffle=True, random_state=seed)
    train_labels = [int(f.stem.split("-")[-1]) for f in train_filepaths]
    train_filepaths, valid_filepaths = train_test_split(train_filepaths, stratify=train_labels, test_size=valid_prop, shuffle=True, random_state=seed)

    train = SimulatedDataset(train_filepaths, num_pos_edges, num_neg_edges)
    valid = SimulatedDataset(valid_filepaths, num_pos_edges, num_neg_edges)
    test = SimulatedDataset(test_filepaths, num_pos_edges, num_neg_edges)

    return {"train": train, "valid": valid, "test": test}
    

class SimulatedDataset(Dataset):


    def __init__(self, filepaths, num_pos_edges=100, num_neg_edges=100):

        self.filepaths = filepaths
        self.num_pos_edges = num_pos_edges
        self.num_neg_edges = num_neg_edges
        self.num_samples = len(self)

        self._get_data_dims()


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):

        graph_timeseries, community_timeseries, label = self._load(self.filepaths[idx])
         
        pos_edges, pos_edge_weights, neg_edges = self._sample_edges(graph_timeseries)

        pos_edges = pos_edges.long()
        pos_edge_weights = pos_edge_weights.float()
        neg_edges = neg_edges.long()
        subject_idx = torch.tensor([idx]).long()
        label = torch.tensor(label).long()

        data = dict(subject_idx=subject_idx, 
                    label=label, 
                    pos_edges=pos_edges, 
                    pos_edge_weights=pos_edge_weights, 
                    neg_edges=neg_edges)

        return data


    def _load(self, filepath):

        data_dict = np.load(filepath, allow_pickle=True)
        graph_timeseries = [data_dict[key].tolist() for key in data_dict.keys() if "g" in key]
        community_timeseries = [data_dict[key] for key in data_dict.keys() if "c" in key]
        label = data_dict["label"]

        return graph_timeseries, community_timeseries, label


    def _get_data_dims(self):

        graph_timeseries, _, _ = self._load(self.filepaths[0])
        
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

        pos_edges = torch.tensor(pos_edges)
        pos_edge_weights = torch.tensor(pos_edge_weights)
        neg_edges = torch.tensor(neg_edges)

        return pos_edges, pos_edge_weights, neg_edges