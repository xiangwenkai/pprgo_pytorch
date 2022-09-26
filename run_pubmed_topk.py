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
    # data_file = config['data_file']  # Path to the .npz data file
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

    data_file = 'data/pubmed.npz'
    data_name = 'pubmed'
    model_save_path = 'model/pubmed/topk/'

    acc_tests = []
    stds = []
    f1_tests = []
    gpu_memorys = []
    memorys = []
    time_total = []
    topks = []
    for topk in range(1, 101, 5):
        topks.append(topk)
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

        # compute the ppr vectors for train/val nodes using ACL's ApproximatePR
        a = adj_matrix.tocoo()

        start = time.time()
        topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                         normalization=ppr_normalization)
        topk_train = normalize(topk_train, norm='l1', axis=1)  # l1 normalize

        # print(time.time() - start)

        train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
        if run_val:
            topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                           normalization=ppr_normalization)
            topk_val = normalize(topk_val, norm='l1', axis=1)  # l1 normalize
            val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
        else:
            val_set = None
        time_preprocessing = time.time() - start

        # 训练n次取平均
        tmp_acc_test = []
        f1_test = 0
        gpu_memory = 0
        memory = 0
        time_train = 0
        n_train = 10
        for i in range(n_train):
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

            torch.save(model, model_save_path + f'model_{topk}_{i}.pt')

            start = time.time()
            predictions, time_logits, time_propagation = predict(
                model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
                nprop=nprop_inference, inf_fraction=inf_fraction,
                ppr_normalization=ppr_normalization)
            time_inference = time.time() - start

            acc = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
            tmp_acc_test.append(acc)

            f1_test += f1_score(labels[test_idx], predictions[test_idx], average='macro')/n_train
            gpu_memory += torch.cuda.max_memory_allocated()/n_train
            memory += utils.get_max_memory_bytes()/n_train
            time_train += (time_training + time_inference)/n_train
        acc_test = np.mean(tmp_acc_test)
        std = np.std(tmp_acc_test)
        print(f'{acc_test};{std};{f1_test};{time_train}')
        acc_tests.append(acc_test)
        stds.append(std)
        f1_tests.append(f1_test)
        gpu_memorys.append(gpu_memory/2**30)
        memorys.append(memory/2**30)
        time_total.append(time_train)


    df = pd.DataFrame({'k':topks, 'gpu_memory':gpu_memorys, 'memory':memorys, 'time':time_total,
                       'acc':acc_tests, 'std':stds, 'f1': f1_tests})
    df.to_csv(f'{data_name}_acc_topk_accurate.csv', index=False)

'''
64.75940599254866;0.42063069117552043;0.6467191547484441;37.24167215824128
66.69885081597315;0.1979715778567731;0.6683963429475291;32.39987509250641
66.80537335362334;0.24004790250618288;0.6694648575515966;31.40369987487793
67.2514036836858;0.28069946331901796;0.6743074390894438;30.41247835159302
67.17374193209847;0.30066409543437894;0.6729204026529753;30.43014616966247
67.30282835703416;0.3652853389909059;0.6743665754092935;30.278347349166868
67.75305661961485;0.3278358359996783;0.6793850506603705;30.446458411216735
67.58408983575589;0.29354823575619643;0.6771351738770814;30.455631446838378
68.20958178097287;0.29685767067799823;0.6842229573485085;30.29823424816131
68.15395917510625;0.3154953838084631;0.683725990980393;30.470617103576654
67.93409245946373;0.34965117723168615;0.6810890179842441;30.396153068542485
68.02277378391143;0.24482947889131595;0.6822210058122835;30.492936468124395
68.07367371569502;0.25769886751095894;0.6825839152198674;30.50641198158264
68.00703153696803;0.2362205620573568;0.6817570831193852;30.506797647476194
68.30980741984573;0.4795276651508741;0.6852638974057211;30.43669846057892
68.0972870861101;0.42884614739304433;0.6827125967857154;30.36608030796051
68.15553339980059;0.4844714683594814;0.6834675866362558;30.529756331443785
68.47195256336254;0.3220053855429467;0.6867323412672104;30.538598775863647
68.27360025187595;0.373544130044933;0.6847480891930391;30.51802628040314
68.28986724038411;0.21070840554969342;0.6847980757383811;30.351930880546572
'''

