import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .dataset import _zscore_timeseries


def _plot_graph(G, tick_prop=0.25, ax=None, figsize=(50, 8), **kwargs):
    
    if isinstance(G, nx.Graph): G = nx.to_numpy_matrix(G)
    num_nodes = int(G.shape[0])
    
    if ax is None: _, ax = plt.subplots(figsize=figsize)
    ax.imshow(G, cmap="gray", vmin=0, vmax=1, interpolation="none", aspect="equal", **kwargs)

    tick_positions = np.arange(0, num_nodes + 1, num_nodes * tick_prop)
    tick_positions[-1] = num_nodes - 0.5
    tick_labels = np.arange(0, num_nodes + 1, num_nodes * tick_prop).astype(int)
    tick_labels[0] = 1

    ax.set_xticks(tick_positions, minor=False); ax.set_yticks(tick_positions, minor=False)
    ax.set_xticklabels(tick_labels); ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Node"); ax.set_xlabel("Node")

def plot_dynamic_graph(G, time_len=None, figsize=(30, 8), title=True, **kwargs):
    ncols  = time_len if time_len is not None else len(G)
    _, axes = plt.subplots(1, ncols, figsize=figsize)
    for idx, (ax, g) in enumerate(zip(axes.ravel(), G)):
        _plot_graph(g, ax=ax, **kwargs)
        if title: ax.set_title("$t={}$".format(idx + 1)) 

def plot_timeseries(X, num_features=None, time_len=None, figsize=(10, 30), 
                    title=None, zscore=True, **kwargs):
    if zscore: X = _zscore_timeseries(X)

    _num_features,  _time_len = X.shape
    num_features  = num_features if num_features is not None else _num_features
    time_len = time_len if time_len is not None else _time_len
    X = X[:num_features, :time_len]

    fig, axes = plt.subplots(num_features, 1, figsize=figsize, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time")
    plt.ylabel("z-score" if zscore else "BOLD signal")

    axes = axes.ravel()
    for idx, (ax, x) in enumerate(zip(axes, X)):
        ax.plot(x)
        if idx < (len(axes) - 1):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
        else:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if title: ax.set_title("...") 
