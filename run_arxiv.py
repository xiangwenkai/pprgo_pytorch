import os
import time
import logging
import yaml
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from pprgo import utils, ppr
from pprgo.pprgo import PPRGo
from pprgo.train import train
from pprgo.predict import predict
from pprgo.dataset import PPRDataset
from ogb.nodeproppred import PygNodePropPredDataset
from pprgo.io import networkx_to_sparsegraph
import pandas as pd
import networkx as nx

# Set up logging
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
        fmt='%(asctime)s (%(levelname)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

with open('config_demo.yaml', 'r') as c:
    config = yaml.safe_load(c)

# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

data_file           = config['data_file']           # Path to the .npz data file
split_seed          = config['split_seed']          # Seed for splitting the dataset into train/val/test
ntrain_div_classes  = config['ntrain_div_classes']  # Number of training nodes divided by number of classes
attr_normalization  = config['attr_normalization']  # Attribute normalization. Not used in the paper

alpha               = config['alpha']               # PPR teleport probability
eps                 = config['eps']                 # Stopping threshold for ACL's ApproximatePR
topk                = config['topk']                # Number of PPR neighbors for each node
ppr_normalization   = config['ppr_normalization']   # Adjacency matrix normalization for weighting neighbors

hidden_size         = config['hidden_size']         # Size of the MLP's hidden layer
nlayers             = config['nlayers']             # Number of MLP layers
weight_decay        = config['weight_decay']        # Weight decay used for training the MLP
dropout             = config['dropout']             # Dropout used for training

lr                  = config['lr']                  # Learning rate
max_epochs          = config['max_epochs']          # Maximum number of epochs (exact number if no early stopping)
batch_size          = config['batch_size']          # Batch size for training
batch_mult_val      = config['batch_mult_val']      # Multiplier for validation batch size

eval_step           = config['eval_step']           # Accuracy is evaluated after every this number of steps
run_val             = config['run_val']             # Evaluate accuracy on validation set during training

early_stop          = config['early_stop']          # Use early stopping
patience            = config['patience']            # Patience for early stopping

nprop_inference     = config['nprop_inference']     # Number of propagation steps during inference
inf_fraction        = config['inf_fraction']        # Fraction of nodes for which local predictions are computed during inference


start = time.time()

graph_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='dataset/')
data = dataset.data

features = data.x.numpy()
label = data.y.squeeze().numpy()

df_edge = pd.DataFrame(data.edge_index.numpy().T, columns=['source', 'target'])
g = nx.from_pandas_edgelist(df_edge, 'source', 'target')
for i in np.arange(len(label)):
    g.nodes[i]['label'] = label[i]
# for i,attr in enumerate(features):
#     for j, value in enumerate(attr):
#         g.nodes[i][j]=value

graph = networkx_to_sparsegraph(g, label_name='label')
adj_matrix=graph.adj_matrix
attr_matrix=features
labels=label
split=dataset.get_idx_split()
train_idx=split['train'].numpy()
val_idx=split['valid'].numpy()
test_idx=split['test'].numpy()


try:
    d = attr_matrix.n_columns
except AttributeError:
    d = attr_matrix.shape[1]
nc = labels.max() + 1
time_loading = time.time() - start
print(f"Runtime: {time_loading:.2f}s")



# compute the ppr vectors for train/val nodes using ACL's ApproximatePR
start = time.time()
topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                 normalization=ppr_normalization)
train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
if run_val:
    topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                   normalization=ppr_normalization)
    val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
else:
    val_set = None
time_preprocessing = time.time() - start
print(f"Runtime: {time_preprocessing:.2f}s")



start = time.time()
model = PPRGo(d, int(nc), hidden_size, nlayers, dropout)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

nepochs, _, _ = train(
        model=model, train_set=train_set, val_set=val_set,
        lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
        eval_step=eval_step, early_stop=early_stop, patience=patience)
time_training = time.time() - start
logging.info('Training done.')
print(f"Runtime: {time_training:.2f}s")


start = time.time()
predictions, time_logits, time_propagation = predict(
        model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
        nprop=nprop_inference, inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization)
time_inference = time.time() - start
print(f"Runtime: {time_inference:.2f}s")


acc_train = 100 * accuracy_score(labels[train_idx], predictions[train_idx])
acc_val = 100 * accuracy_score(labels[val_idx], predictions[val_idx])
acc_test = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
f1_val = f1_score(labels[val_idx], predictions[val_idx], average='macro')
f1_test = f1_score(labels[test_idx], predictions[test_idx], average='macro')

gpu_memory = torch.cuda.max_memory_allocated()
memory = utils.get_max_memory_bytes()

time_total = time_preprocessing + time_training + time_inference


print(f'''
Accuracy: Train: {acc_train:.1f}%, val: {acc_val:.1f}%, test: {acc_test:.1f}%
F1 score: Train: {f1_train:.3f}, val: {f1_val:.3f}, test: {f1_test:.3f}

Runtime: Preprocessing: {time_preprocessing:.2f}s, training: {time_training:.2f}s, inference: {time_inference:.2f}s -> total: {time_total:.2f}s
Memory: Main: {memory / 2**30:.2f}GB, GPU: {gpu_memory / 2**30:.3f}GB
''')





