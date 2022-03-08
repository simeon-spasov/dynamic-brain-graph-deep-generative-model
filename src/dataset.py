from collections import OrderedDict
from pathlib import Path
from random import randrange

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset


def load_hcp_datasets(valid_split=0.10, test_split=0.10, num_nodes=200, num_timesteps=1200, 
                      window_size=None, seed=12345, data_dir="./data"):

    # load subject metadata 
    filepath = Path(data_dir) / "hcp_rest" / "metadata.csv"
    df = pd.read_csv(filepath)

    train_idx, test_idx = next(ShuffleSplit(1, test_size=test_split, random_state=seed).split(df.Subject))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]  
    train_idx, valid_idx  = next(ShuffleSplit(1, test_size=valid_split, random_state=seed).split(train_df.Subject))
    train_df, valid_df = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    
    train = HCPDataset(train_df, num_nodes, num_timesteps, window_size, data_dir)
    valid = HCPDataset(valid_df, num_nodes, num_timesteps, window_size, data_dir)
    test = HCPDataset(test_df, num_nodes, num_timesteps, None, data_dir)
    
    return dict(train=train, valid=valid, test=test)

class HCPDataset(Dataset):
    def __init__(self, df, num_nodes=200, num_timesteps=1200, window_size=None, data_dir="./data"):
        super().__init__()

        self.data_dir = Path(data_dir) / "hcp_rest" / "3T_HCP1200_MSMAll_d{num_nodes}_ts2".format(num_nodes=num_nodes)
        self.filepaths = OrderedDict({int(p.stem): p for p in self.data_dir.iterdir()})

        self.sub_ids = df["Subject"].to_numpy()
        self.num_timesteps = num_timesteps
        self.window_size = window_size
                             
    def __len__(self):
        return len(self.sub_ids)
    
    def __getitem__(self, idx):
        sub_id = torch.tensor(self.sub_ids[idx]).type(torch.int32)       
        x = pd.read_csv(self.filepaths[self.sub_ids[idx]], delimiter=" ").to_numpy().T
        x = x[:, :self.num_timesteps]
        x = stats.zscore(x, axis=-1)
        x = torch.from_numpy(x).type(torch.float32)
        
        if self.window_size is not None:
            sampling_idx = randrange(x.shape[-1] - self.window_size)
            x = x[:, sampling_idx: sampling_idx + self.window_size]
    
        return dict(id=sub_id, x=x) 
