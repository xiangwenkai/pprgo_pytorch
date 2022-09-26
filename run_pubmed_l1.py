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

    data_file = 'data/pubmed.npz'
    data_name = 'pubmed'
    model_save_path = 'model/pubmed/l1/'

    acc_tests = []
    stds = []
    f1_tests = []
    gpu_memorys = []
    memorys = []
    nonzero_num = []
    time_total = []
    epss = [0.05, 0.03, 0.01, 0.008, 0.006,
            0.005, 0.004, 0.003, 0.002, 0.001,
            0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
            0.00009, 0.00008, 0.00007, 0.00006, 0.00005]

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

    for eps in epss:
        # compute the ppr vectors for train/val nodes using ACL's ApproximatePR

        start = time.time()
        l1_train = ppr.l1_ppr_matrix(adj_matrix, alpha, eps, train_idx,
                                     normalization=ppr_normalization)
        l1_train = normalize(l1_train, norm='l1', axis=1)  # l1 normalize
        adj_degree = np.sum(l1_train > 0, axis=1).A1
        avg_num = np.mean(adj_degree)
        nonzero_num.append(avg_num)

        # print(time.time() - start)

        train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=l1_train, indices=train_idx, labels_all=labels)
        if run_val:
            l1_val = ppr.l1_ppr_matrix(adj_matrix, alpha, eps, val_idx,
                                           normalization=ppr_normalization)
            l1_val = normalize(l1_val, norm='l1', axis=1)  # l1 normalize
            val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=l1_val, indices=val_idx, labels_all=labels)
        else:
            val_set = None
        time_preprocessing = time.time() - start

        # 训练n次取平均
        tmp_acc_test = []
        std = 0
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

            start = time.time()
            predictions, time_logits, time_propagation = predict(
                model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
                nprop=nprop_inference, inf_fraction=inf_fraction,
                ppr_normalization=ppr_normalization)
            time_inference = time.time() - start

            torch.save(model, model_save_path + f'model_{eps}_{i}.pt')

            acc = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
            tmp_acc_test.append(acc)

            f1_test += f1_score(labels[test_idx], predictions[test_idx], average='macro')/n_train
            gpu_memory += torch.cuda.max_memory_allocated() / n_train
            memory += utils.get_max_memory_bytes() / n_train
            time_train += (time_training + time_inference) / n_train
        acc_test = np.mean(tmp_acc_test)
        std = np.std(tmp_acc_test)
        print(f'{acc_test};{std};{f1_test};{time_train}')
        acc_tests.append(acc_test)
        stds.append(std)
        f1_tests.append(f1_test)
        gpu_memorys.append(gpu_memory/2**30)
        memorys.append(memory/2**30)
        time_total.append(time_train)

    df = pd.DataFrame({'eps': epss, 'gpu_memory': gpu_memorys, 'memory': memorys, 'time': time_total,
                       'acc': acc_tests, 'f1': f1_tests, 'std':stds, 'nonzero_num': nonzero_num})
    df.to_csv(f'{data_name}_acc_l1.csv', index=False)


'''
64.7662276328908;0.3241293433312391;0.6459126789287662;37.258244609832765
65.25371254657082;0.4313694836544005;0.651713166438264;36.16174364089966
65.950044603033;0.32968180798830654;0.6594037480955248;36.2390460729599
65.93902503017264;0.3622085429344251;0.6598846043361513;36.24173233509064
66.17305976806423;0.32358476958299165;0.661905939903898;32.25156774520874
66.23392978957864;0.2156145146992392;0.6625381440703919;32.33462617397309
66.63063441255181;0.22139865854385046;0.6671257995606158;32.33885633945465
66.8163929264837;0.23945470368190122;0.6695997387596482;31.346092271804807
67.01841842892375;0.24014653096745534;0.6713903793952618;30.368097829818726
67.50695282573334;0.4227398293562489;0.6767120834407356;30.37020246982575
67.73469066484755;0.357069573576676;0.6792210213072425;30.206864476203922
67.67487012646271;0.43344923792841983;0.6784609929220426;30.37456021308899
67.59353518392192;0.39539233380754407;0.6774904406229724;30.382688260078428
67.89001416802225;0.31588221846924286;0.6809620374273784;30.299725651741028
68.02224904234666;0.2239853698946643;0.6824280269424585;30.38952713012695
68.03851603085482;0.39674383783180117;0.6823260671733342;30.39376001358032
68.31715380175265;0.5087709769890565;0.6852945998091394;30.413863134384155
68.43994332791101;0.27538885266093177;0.6863282544660939;30.46403589248657
68.44833919294747;0.3690153414604217;0.6865558807597862;30.37587203979492
68.3869444298683;0.28779814571764767;0.6855874108167643;30.305333900451664
68.49189274282416;0.24656432998257277;0.6870543769317137;30.46319944858551
68.49556593377761;0.45197767189434307;0.6867933464889755;30.481526494026188
68.52180301201659;0.4106873019590294;0.687184745433889;30.498595595359802
'''