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
from pprgo.utils import SparseRowIndexer, split_random, clustering_coefficient
import seaborn as sns


if __name__ == "__main__":
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

    cluster_coef = clustering_coefficient(adj_matrix, test_idx)
    sns.distplot(cluster_coef)
    plt.savefig(f'plot/{data_name}/{data_name}_cluster_coef_dis.png', dpi=300)
    plt.close()
    # plt.show()

    cluster_coef_group = pd.cut(cluster_coef, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    model_names = os.listdir(model_path)
    if method == 'l1':
        epss = sorted(list(set([float(x.split('_')[1]) for x in model_names])))[::-1]
    if method == 'topk':
        epss = sorted(list(set([int(x.split('_')[1]) for x in model_names])))

    eps = 0.01
    i = 0

    model = torch.load(f'model/{data_name}/{method}/model_{eps}_{i}.pt')

    predictions, time_logits, time_propagation = predict(
        model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
        nprop=nprop_inference, inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization)

    cat_report = classification_report(labels[test_idx], predictions[test_idx], output_dict=True)

    df = pd.DataFrame({'node_idx': test_idx,
                       'label': labels[test_idx],
                       'prediction': predictions[test_idx],
                       'cluster_coef': cluster_coef,
                       'cluster_coef_group': cluster_coef_group})
    df['prediction'] = predictions[test_idx]
    df['acc'] = df.apply(lambda x: 1 if x['label'] == x['prediction'] else 0, axis=1)

    df['neighbor_num'] = [len(adj_matrix.indices[adj_matrix.indptr[v]:adj_matrix.indptr[v+1]]) for v in test_idx]

    df['expected_coef'] = 2 * 114615892 / (232965*232964)



    df['test'] = df.apply(
        lambda x: x['cluster_coef'], axis=1)
    df['test_group'] = pd.qcut(df['test'], q=[0,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1], duplicates='drop')
    df[['test_group', 'acc']].groupby('test_group').mean()



    df.to_csv(f'result/{data_name}_cluster_coef.csv', index=False)



    # df = pd.read_csv(f'result/{data_name}_cluster_coef.csv')
    res = pd.DataFrame({'id': list(range(18))})
    n_train = 10
    for eps in epss:
        avg_acc = np.zeros(len(test_idx))
        for i in range(n_train):
            model = torch.load(f'model/{data_name}/{method}/model_{eps}_{i}.pt')

            predictions, time_logits, time_propagation = predict(
                model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
                nprop=nprop_inference, inf_fraction=inf_fraction,
                ppr_normalization=ppr_normalization)
            acc = labels[test_idx] == predictions[test_idx]
            acc = acc.astype(float)
            avg_acc += acc/n_train
        df['acc'] = avg_acc

        # df['test'] = df.apply(
        #     lambda x: x['cluster_coef'], axis=1)
        # df['test_group'] = pd.qcut(df['test'], q=np.linspace(0,1,21),
        #                            duplicates='drop')
        # tmp = df[['test_group', 'acc']].groupby('test_group').mean().reset_index()
        # res[f'acc_{eps}'] = tmp['acc']

        # df['test'] = df.apply(
        #     lambda x: x['neighbor_num']/(x['cluster_coef'] + 1e-5), axis=1)
        # df['test_group'] = pd.qcut(df['test'], q=np.linspace(0, 1, 21),
        #                            duplicates='drop')
        # tmp = df[['test_group', 'acc']].groupby('test_group').mean().reset_index()
        # res[f'acc_{eps}'] = tmp['acc']

        df['group'] = pd.qcut(df['neighbor_num'], q=np.linspace(0, 1, 21),
                                   duplicates='drop')
        tmp = df[['group', 'acc']].groupby('group').mean().reset_index()
        res[f'acc_{eps}'] = tmp['acc']

    plt.style.use('ggplot')
    res = res.T
    for i in range(18):
        plt.plot(res[i][1:])
    plt.show()
    plt.close()

