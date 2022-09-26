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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp
from pprgo.utils import SparseRowIndexer, split_random


if __name__=="__main__":
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

    # data_file = 'data/cora_full.npz'
    # data_name = 'cora'
    data_name = 'reddit'
    method = 'l1'
    model_path = f'model/{data_name}/{method}'
    if data_name in ['pubmed', 'cora']:
        file_map = {'pubmed': 'pubmed.npz', 'cora': 'cora_full.npz'}
        (adj_matrix, attr_matrix, labels,
         train_idx, val_idx, test_idx) = utils.get_data(
            f"data/{file_map[data_name]}",
            seed=split_seed,
            ntrain_div_classes=ntrain_div_classes,
            normalize_attr=attr_normalization
        )
        try:
            d = attr_matrix.n_columns
        except AttributeError:
            d = attr_matrix.shape[1]
        nc = labels.max() + 1
    if data_name in ['arxiv']:
        dataset = PygNodePropPredDataset(name=f'ogbn-arxiv')
        data = dataset[0]
        attr_matrix = data.x.numpy()
        n = len(attr_matrix)

        attr_matrix = sp.csr_matrix(attr_matrix)
        attr_matrix = SparseRowIndexer(attr_matrix)

        edges = data.edge_index.numpy().T

        adj_matrix = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                   shape=(n, n),
                                   dtype=np.float32)

        labels = data.y.numpy().reshape(-1)
        ###############read ogb#####################
        try:
            d = attr_matrix.n_columns
        except AttributeError:
            d = attr_matrix.shape[1]
        nc = labels.max() + 1

        n_train = nc * ntrain_div_classes
        n_val = n_train * 10
        train_idx, val_idx, test_idx = split_random(split_seed, n, n_train, n_val)

    if data_name in ['reddit']:
        data_file = 'data/Reddit'
        (adj_matrix, attr_matrix, labels,
         train_idx, val_idx, test_idx) = utils.get_reddit(
            f"{data_file}",
            seed=split_seed,
            ntrain_div_classes=ntrain_div_classes,
        )
        try:
            d = attr_matrix.n_columns
        except AttributeError:
            d = attr_matrix.shape[1]
        nc = labels.max() + 1

    model_names = os.listdir(model_path)
    epss = sorted(list(set([float(x.split('_')[1]) for x in model_names])))[::-1]
    # if data_name == 'pubmed':
    #     epss = [0.05, 0.03, 0.01, 0.008, 0.006,
    #             0.005, 0.004, 0.003, 0.002, 0.001,
    #             0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
    #             0.00009, 0.00008, 0.00007, 0.00006, 0.00005]
    # if data_name == 'cora':
    #     epss = [0.05, 0.03, 0.01, 0.005, 0.003, 0.001,
    #             0.00075, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
    #             0.000075, 0.00005, 0.00003, 0.00001]
    # if data_name == 'arxiv':
    #     epss = [0.05, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001,
    #             0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00007, 0.00005]
    n_train = 10

    for cat in set(labels):
        locals()[f'all_precision_{cat}_mean'] = []
        locals()[f'all_recall_{cat}_mean'] = []
        locals()[f'all_f1_{cat}_mean'] = []
        locals()[f'all_precision_{cat}_std'] = []
        locals()[f'all_recall_{cat}_std'] = []
        locals()[f'all_f1_{cat}_std'] = []
    for eps in epss:
        for cat in set(labels):
            locals()[f'precision_{cat}'] = []
            locals()[f'recall_{cat}'] = []
            locals()[f'f1_{cat}'] = []
        for i in range(n_train):
            # model = PPRGo(d, nc, hidden_size, nlayers, dropout)
            # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            # model.to(device)
            model = torch.load(f'model/{data_name}/{method}/model_{eps}_{i}.pt')

            predictions, time_logits, time_propagation = predict(
                model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
                nprop=nprop_inference, inf_fraction=inf_fraction,
                ppr_normalization=ppr_normalization)
            cat_report = classification_report(labels[test_idx], predictions[test_idx], output_dict=True)
            for cat in set(labels):
                cat = str(cat)
                locals()[f'precision_{cat}'].append(cat_report[cat]['precision'])
                locals()[f'recall_{cat}'].append(cat_report[cat]['recall'])
                locals()[f'f1_{cat}'].append(cat_report[cat]['f1-score'])
        for cat in set(labels):
            locals()[f'all_precision_{cat}_mean'].append(np.mean(locals()[f'precision_{cat}']))
            locals()[f'all_recall_{cat}_mean'].append(np.mean(locals()[f'recall_{cat}']))
            locals()[f'all_f1_{cat}_mean'].append(np.mean(locals()[f'f1_{cat}']))
            locals()[f'all_precision_{cat}_std'].append(np.std(locals()[f'precision_{cat}']))
            locals()[f'all_recall_{cat}_std'].append(np.std(locals()[f'recall_{cat}']))
            locals()[f'all_f1_{cat}_std'].append(np.std(locals()[f'f1_{cat}']))
    df = pd.DataFrame({f'{method}': epss})
    for cat in set(labels):
        df[f'f1_{cat}_mean'] = locals()[f'all_f1_{cat}_mean']
        df[f'f1_{cat}_std'] = locals()[f'all_f1_{cat}_std']

    df.to_csv(f'result/{data_name}_{method}_cat_acc.csv', index=False)

    plt.style.use('ggplot')
    a, b = np.unique(labels, return_counts=True)
    intervals = [0, 2500, 3500, 4500, 10000, max(b)]
    for i in range(len(intervals)-1):
        for cat in set(labels):
            if intervals[i]<=b[cat]<intervals[i+1]:
                plt.errorbar(df.index, df[f'f1_{cat}_mean'], yerr=df[f'f1_{cat}_std'],
                             fmt='o-', label=f'{cat}')
        plt.title(f'{data_name}-epsilon-accuracy')
        plt.legend()
        plt.savefig(f'plot/{data_name}/{method}/cat_{intervals[i]}_{intervals[i+1]}_acc.png', dpi=300)
        plt.close()
    # np.unique(labels, return_counts=True)


