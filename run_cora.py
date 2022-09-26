import os
import time
import logging

import pandas as pd
import yaml
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import networkx as nx
from pprgo import utils, ppr
from pprgo.pprgo import PPRGo
from pprgo.train import train
from pprgo.predict import predict
from pprgo.dataset import PPRDataset
from pprgo import my_ppr
from sklearn.preprocessing import normalize

with open('config_demo.yaml', 'r') as c:
    config = yaml.safe_load(c)
    # For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
data_file = config['data_file']  # Path to the .npz data file
split_seed = config['split_seed']  # Seed for splitting the dataset into train/val/test
ntrain_div_classes = config['ntrain_div_classes']  # Number of training nodes divided by number of classes
attr_normalization = config['attr_normalization']  # Attribute normalization. Not used in the paper

alpha = config['alpha']  # PPR teleport probability
eps = config['eps']  # Stopping threshold for ACL's ApproximatePR
topk = config['topk']  # Number of PPR neighbors for each node
ppr_normalization = config['ppr_normalization']  # Adjacency matrix normalization for weighting neighbors

hidden_size = config['hidden_size']  # Size of the MLP's hidden layer
nlayers = config['nlayers']  # Number of MLP layers
weight_decay = config['weight_decay']  # Weight decay used for training the MLP
dropout = config['dropout']  # Dropout used for training

lr = config['lr']  # Learning rate
max_epochs = config['max_epochs']  # Maximum number of epochs (exact number if no early stopping)
batch_size = config['batch_size']  # Batch size for training
batch_mult_val = config['batch_mult_val']  # Multiplier for validation batch size

eval_step = config['eval_step']  # Accuracy is evaluated after every this number of steps
run_val = config['run_val']  # Evaluate accuracy on validation set during training

early_stop = config['early_stop']  # Use early stopping
patience = config['patience']  # Patience for early stopping
nprop_inference = config['nprop_inference']  # Number of propagation steps during inference
inf_fraction = config['inf_fraction']  # Fraction of nodes for which local predictions are computed during inference

data_file = 'data/cora_full.npz'
data_name = 'cora'

acc_tests = []
f1_tests = []
gpu_memorys = []
memorys = []
nonzero_num = []
time_total = []
epss = [0.05, 0.03, 0.01, 0.005, 0.003, 0.001,
        0.00075, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
        0.000075, 0.00005, 0.00003, 0.00001]
for eps in epss:
    start = time.time()
    (adj_matrix, attr_matrix, labels,
     train_idx, val_idx, test_idx) = utils.get_data(
        f"{data_file}",
        seed=split_seed,
        ntrain_div_classes=ntrain_div_classes,
        normalize_attr=attr_normalization
    )
    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]
    nc = labels.max() + 1
    time_loading = time.time() - start
    print(f"Runtime: {time_loading:.2f}s")

    # compute the ppr vectors for train/val nodes using ACL's ApproximatePR
    a = adj_matrix.tocoo()

    start = time.time()
    l1_train = ppr.l1_ppr_matrix(adj_matrix, alpha, eps, train_idx,
                                     normalization=ppr_normalization)
    l1_train = normalize(l1_train, norm='l1', axis=1)  # l1 normalize
    adj_degree = np.sum(l1_train > 0, axis=1).A1
    avg_num = np.mean(adj_degree)
    nonzero_num.append(avg_num)

    print(time.time() - start)

    train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=l1_train, indices=train_idx, labels_all=labels)
    if run_val:
        l1_val = ppr.l1_ppr_matrix(adj_matrix, alpha, eps, val_idx,
                                       normalization=ppr_normalization)
        l1_val = normalize(l1_val, norm='l1', axis=1)  # l1 normalize
        val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=l1_val, indices=val_idx, labels_all=labels)
    else:
        val_set = None
    time_preprocessing = time.time() - start
    print(f"Runtime: {time_preprocessing:.2f}s")

    # 训练n次取平均
    acc_test = 0
    f1_test = 0
    gpu_memory = 0
    memory = 0
    time_train = 0
    n_train = 3
    for _ in range(n_train):
        start = time.time()
        model = PPRGo(d, nc, hidden_size, nlayers, dropout)
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

        # acc_train = 100 * accuracy_score(labels[train_idx], predictions[train_idx])
        # acc_val = 100 * accuracy_score(labels[val_idx], predictions[val_idx])
        acc_test += 100 * accuracy_score(labels[test_idx], predictions[test_idx])/n_train
        # f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
        # f1_val = f1_score(labels[val_idx], predictions[val_idx], average='macro')
        f1_test += f1_score(labels[test_idx], predictions[test_idx], average='macro')/n_train
        gpu_memory += torch.cuda.max_memory_allocated() / n_train
        memory += utils.get_max_memory_bytes() / n_train
        time_train += (time_training + time_inference) / n_train
    acc_tests.append(acc_test)
    f1_tests.append(f1_test)
    gpu_memorys.append(gpu_memory)
    memorys.append(memory)
    time_total.append(time_train)

df = pd.DataFrame({'eps': epss, 'gpu_memory': gpu_memorys, 'memory': memorys, 'time': time_total,
                   'acc': acc_tests, 'f1': f1_tests, 'nonzero_num': nonzero_num})
df.to_csv(f'{data_name}_acc_l1.csv', index=False)
