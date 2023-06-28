import logging
import pathlib
import pickle

import networkx as nx
import nilearn.connectome as conn
import numpy as np


def _get_filepaths(dataset="hcp", data_dir="./data", ext=".npy"):
    data_dir = pathlib.Path(data_dir) / dataset / "raw"
    return sorted(list(data_dir.rglob("*" + ext)))


def _zscore_timeseries(X, axis=-1):
    X -= np.mean(X, axis=axis, keepdims=True)
    X /= np.std(X, axis=axis, keepdims=True)
    return X


def _load_timeseries(filepath, zscore=False, dtype=np.float64):
    # load multivariate timeseries (nodes x time length)
    X = np.load(filepath).astype(dtype)
    # zscore timeseries
    if zscore: X = _zscore_timeseries(X)
    # gender label
    y = int(filepath.stem.split("_")[-1])
    return X, y


def _compute_dynamic_fc(X,
                        window_size=30,
                        window_stride=10,
                        measure="correlation",
                        top_percent=5,
                        max_time=490,
                        zscore=True,
                        self_loops=False,
                        as_graph=True):
    # truncate time length
    time_len = X.shape[-1]
    max_time = max_time if (max_time and max_time <= time_len) else time_len
    X = X[:, :max_time]

    if zscore: X = _zscore_timeseries(X)

    # calculate starting timepoint for each window
    sampling_points = list(range(0, max_time - window_size, window_stride))
    # initialize functional connectivity measure (uses Ledoit-Wolf covariance estimator)
    conn_measure = conn.ConnectivityMeasure(kind=measure)

    # calculate dynamic functional connectivity within each timeseries window 
    G = []
    for idx in sampling_points:
        # calculate functional connectivity matrix
        A = conn_measure.fit_transform([X[:, idx: idx + window_size].T])[0]
        # remove self-loops
        if not self_loops: np.fill_diagonal(A, 0.)
        # calculate percentile threshold value
        threshold = np.percentile(A.flatten(), 100. - top_percent)
        # threshold 
        A[A < threshold] = 0.
        # return as networkx graph
        G += [nx.from_numpy_matrix(A) if as_graph else A]

    return G


def load_dataset(dataset="hcp", window_size=30, window_stride=10, measure="correlation",
                 top_percent=5, data_dir="../data", **kwargs):
    # load dataset if already exists
    filename = "{}_w-{}_s-{}_m-{}_p-{}.pkl".format(dataset, window_size, window_stride, measure, top_percent)
    _filepath = pathlib.Path(data_dir) / dataset / filename
    if _filepath.exists():
        logging.info('Found an existing .pkl dataset. Loading...')
        with open(_filepath, "rb") as input_file:
            _dataset = pickle.load(input_file)
    # create and save dataset if does not exist
    else:
        logging.info('No existing .pkl dataset. Pre-processing fMRI timeseries...')
        _dataset = []
        for idx, filepath in enumerate(_get_filepaths(dataset, data_dir)):
            logging.info(f'Loading timeseries for file {filepath}. Index: {idx}.')
            X, y = _load_timeseries(filepath)
            logging.info(f'Computing dynamic functional connectivitiy...')
            G = _compute_dynamic_fc(X, window_size, window_stride, measure, top_percent, **kwargs)
            _dataset += [[idx, G, y]]
        with open(_filepath, "wb") as output_file:
            pickle.dump(_dataset, output_file)
    logging.info('Loaded dataset.')
    return _dataset


def data_loader(dataset, batch_size=10):
    num_samples = len(dataset)
    sample_idx = list(range(num_samples))
    if num_samples < batch_size:
        num_batches = 1
    else:
        num_batches = num_samples // batch_size
    batch_sample_idx = np.array_split(sample_idx, num_batches)
    for batch in batch_sample_idx:
        yield list(map(lambda x: dataset[x], batch))
