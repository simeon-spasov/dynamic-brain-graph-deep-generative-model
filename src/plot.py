import matplotlib.pyplot as plt
import numpy as np


def _plot_graph(graph, label=None, tick_prop=0.25, ax=None, figsize=(50, 8), **kwargs):
    
    # if isinstance(graph, sp.csr_matrix): 
    #     graph = graph.toarray()
    #     graph = (graph + graph.T) / 2
    #     np.fill_diagonal(graph, 1.0)
        
    if ax is None: _, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(graph, cmap="gray", vmin=graph.min(), vmax=graph.max(), interpolation="none", aspect="equal", **kwargs)

    num_nodes = int(graph.shape[0])
    tick_positions = np.arange(0, num_nodes + 1, num_nodes * tick_prop)
    tick_positions[-1] = num_nodes - 0.5
    tick_labels = np.arange(0, num_nodes + 1, num_nodes * tick_prop).astype(int)
    tick_labels[0] = 1

    ax.set_xticks(tick_positions, minor=False); ax.set_yticks(tick_positions, minor=False)
    ax.set_xticklabels(tick_labels); ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Node"); ax.set_xlabel("Node")


def plot_dynamic_graph(dynamic_graph, nrows=None, ncols=None, figsize=(30, 8), **kwargs):
    
    nrows = nrows if nrows is not None else 1
    ncols  = ncols if ncols is not None else len(dynamic_graph)

    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for idx, (ax, graph) in enumerate(zip(axes.ravel(), dynamic_graph)):
        _plot_graph(graph, ax=ax, **kwargs)
        ax.set_title("$t={}$".format(idx + 1)) 
