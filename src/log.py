import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

METRICS = ["elbo", "nll", "kl_alpha", "kl_beta", "kl_phi", "kl_z"]

class Logger:
    def __init__(self, split="train", results_dir="./results"):
        super().__init__()
        self.metrics = METRICS
        self.split = split
        self.results_dir = Path(results_dir)
        self._initialize()

    def _initialize(self):
        self.data = OrderedDict({m: list() for m in self.metrics})

    def update(self, metrics):
        for key, value in metrics.items():
            self.data[key] += [value] 
 
    def save(self):
        filepath = self.results_dir / "{}_metrics.csv".format(self.split)
        exists = filepath.is_file()
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(self.data.keys())
            writer.writerows(zip(* self.data.values()))
