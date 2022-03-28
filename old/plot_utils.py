import numpy 
import matplotlib.pyplot as plt
import seaborn
import pathlib

ROI_LABELS = numpy.genfromtxt("roi_labels.txt", dtype="str")[:-1]

def plot_community_node_distributions(node_dist, figsize=(15, 30), cmap="YlGnBu", save_dir="./figs", filename="node_community.png"):
    node_dist = numpy.array(node_dist).swapaxes(1, 2).swapaxes(0, 2)
    num_community, num_roi, time_len = node_dist.shape
    
    fig, axes = plt.subplots(num_community, 1, figsize=figsize, gridspec_kw = {"wspace":0, "hspace":0.10})
    for idx, ax in enumerate(axes):
        seaborn.heatmap(node_dist[idx, ...], linewidth=0.1, ax=ax, cmap=cmap)
        ax.set_ylabel("ROI")
        ax.set_xlabel("Time")
        ax.set_title("Community {}".format(idx))
    
    if save_dir and filename:
        plt.savefig(pathlib.Path(save_dir).joinpath(filename), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()