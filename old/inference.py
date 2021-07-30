import pickle as pkl
import numpy as np
import torch

#NB! Change paths to your destination directory
import sys
sys.path.append('/Users/simeonspasov/Documents/fMRI')
from data_utils_temp import load_fmri_graphs, get_path_to_files
from dynmodel_hmm_batched import EvolveGraph

save_path = '/Users/simeonspasov/Documents/fMRI/' 
embedding_dim = 32
epochs = 10001
temp_min = 0.1
ANNEAL_RATE = 0.00003
lamda = 100
DATA_DIR = "/Users/simeonspasov/Documents/fMRI/fMRI_data_small"

window_size = 1
stride = 1

categorical_dim = 3  # the number of established brain functional networks
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_nodes = 376
gamma = 0.1  #gamma is 0.1
sigma = 1.  #sigma is 1.
num_batches = 100

path_to_files_list = get_path_to_files(DATA_DIR)
num_subjects = len(path_to_files_list)
model = EvolveGraph(num_subjects, num_nodes, embedding_dim, categorical_dim,
                    gamma, sigma, device)

loss_ = []
temp = 1.

#Need to load all the data in batches
files = path_to_files_list
graphs = [[
    batch_idx,
    load_fmri_graphs(path_to_file, window_size=window_size, stride=stride)]
     for batch_idx, path_to_file in files]

model_path = save_path + '/fmri_model.pt'
model.load_state_dict(torch.load(model_path))
model.eval()


batch_subject_distrib_over_nodes = model.inference(graphs)

with open(save_path + 'static_comms_fmri.pkl', 'wb') as f:
    pkl.dump(batch_subject_distrib_over_nodes, f)


model.train()    
graph_batches = np.array_split(graphs, num_batches)
for batch_graphs in graph_batches:
    loss = model(batch_graphs)
    print(loss)
    loss_.append(loss)
    
print ('Per subject loss is ', np.mean(loss_))

#with open(save_path + 'static_comms_top_10_percent.pkl', 'rb') as f:
#    res = pkl.load(f)
