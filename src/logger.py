import csv
from pathlib import Path


def save_results(metrics, split, results_dir):
    filepath = Path(results_dir) / "{}_metrics.csv".format(split)
    exists = filepath.is_file()
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(metrics.keys())
        writer.writerow(list(metrics.values()))
