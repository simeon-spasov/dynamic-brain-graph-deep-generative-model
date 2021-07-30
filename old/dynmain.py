import pickle as pkl
import numpy as np
import torch
import random

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
cur_lr = 0.005
DATA_DIR = "/Users/simeonspasov/Documents/fMRI/fMRI_data_small"

window_size = 1
stride = 1

categorical_dim = 3  # the number of established brain functional networks
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_nodes = 376
gamma = 0.1  #gamma is 0.1
sigma = 1.  #sigma is 1.
num_batches = 517

path_to_files_list = get_path_to_files(DATA_DIR)
num_subjects = len(path_to_files_list)
model = EvolveGraph(num_subjects, num_nodes, embedding_dim, categorical_dim,
                    gamma, sigma, device)

loss_ = []
temp = 1.

#Need to load all the data in batches
batches = []
file_batches = np.array_split(path_to_files_list, num_batches)
for file_batch in file_batches:
        batch_graphs = [[
            batch_idx,
            load_fmri_graphs(path_to_file, window_size=window_size, stride=stride)
        ] for batch_idx, path_to_file in file_batch]
        batches.append(batch_graphs)

for epoch in range(epochs):
    print('Epoch is ', epoch)
    random.shuffle(batches)
    for batch_graphs in batches:
        model.train()
        loss = model(batch_graphs)
        #print(loss)
        loss_.append(loss)
    if epoch%10 == 0:
        torch.save(model.state_dict(), save_path + '/fmri_mod.pt')
    if epoch % 100 == 0:
        temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
        cur_lr *= .99
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = cur_lr

#plt.plot(means)
#plt.xlabel('Epochs')
#plt.ylabel('Loss: BCE + KLD_z')
#plt.savefig(save_path + "loss_curve.pdf", bbox_inches='tight')

#node_distrib_over_communities = model.inference(graphs)
#with open(save_path + 'static_comms_weighted_top_50_percent.pkl', 'wb') as f:
#    pkl.dump(node_distrib_over_communities, f)

#with open(save_path + 'static_comms_top_10_percent.pkl', 'rb') as f:
#    res = pkl.load(f)

# Check code again
#Use phi and beta for node and community embeddings?
#Incorporate edge weights and edge-weigted training from correlation matrix
#Incorporate node features in NN functions, or send to GRU in inference procedure
#Model might forget subject embedding because of GRU sequential model long sequence issues.
torch.save(model.state_dict(), save_path + '/fmri_mod.pt')
print("Optimization Finished!")
