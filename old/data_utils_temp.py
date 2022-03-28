import pathlib
import numpy as np
import nilearn as nl
import nilearn.connectome as conn
#import matplotlib.pyplot as plt
import networkx as nx


def get_path_to_files(data_dir="./data", ext=".txt"):
    data_dir = pathlib.Path(data_dir)
    path_to_file_list = sorted(
        [path_to_file for path_to_file in data_dir.rglob("*" + ext)])
    return [[idx, path_to_file]
            for idx, path_to_file in enumerate(path_to_file_list)]


def load_data(path_to_file):
    data = np.loadtxt(path_to_file, delimiter=',').T
    noise = np.random.normal(0, 1, data.shape)
    return data + noise


def create_dynamic_connectivity_matrix(data,
                                       window_size,
                                       stride,
                                       measure="correlation"):
    max_time = data.shape[0]
    correlation_measure = conn.ConnectivityMeasure(
        kind=measure)  #  uses Ledoit-Wolf covariance estimator
    #sub_windows = (np.expand_dims(np.arange(window_size), 0) + np.expand_dims(np.arange(max_time, step=stride), 0).T)
    sub_windows = [np.arange(i, i + 480) for i in range(5)]
    windowed_data = data[sub_windows, :]
    return np.array([
        correlation_measure.fit_transform([window])[0]
        for window in windowed_data
    ])


#path_to_file = "/Users/simeonspasov/Downloads/UKB1000211_ts_raw.txt"


def load_fmri_graphs(path, window_size, stride, top_percent=2):

    #window_size = 10
    #stride = 10

    data = load_data(path)
    #print(data.shape)

    d_conn = create_dynamic_connectivity_matrix(data, window_size, stride)
    #print(d_conn.shape)

    graphs = []
    time_steps, node_num, _ = d_conn.shape
    for t in range(time_steps):
        corr_matrix = d_conn[t]
        np.fill_diagonal(corr_matrix, 0.)
        threshold = np.percentile(
            corr_matrix.flatten(),
            100 - top_percent)  #take top percentile of positive correlations
        corr_matrix[corr_matrix < threshold] = 0.
        G = nx.from_numpy_matrix(
            corr_matrix
        )  #TODO: We can also take the correlation values and as edge weights
        graphs.append(G)
    return graphs


#G = graphs[0]
#pos = nx.drawing.nx_agraph.graphviz_layout(G)
#nx.draw(G, pos, node_size=8)

#fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
#for idx, ax in zip(range(5), axes.ravel()):
#    ax.imshow(d_conn[idx], cmap="coolwarm", vmin=-1, vmax=1)
#plt.show()

#save_path = path_to_file.split('_')[0] + "_w-" + str(window_size) + "_s-" + str(stride) + ".npy"
#np.save(save_path, data)
    
import pickle as pkl
with open('/Users/simeonspasov/Downloads/fMRI_codestatic_comms_fmri.pkl', "rb") as f:
    subjects = pkl.load(f)
