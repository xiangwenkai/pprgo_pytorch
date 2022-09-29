import resource
import numpy as np
import scipy.sparse as sp
import sklearn
from torch_geometric.datasets import Reddit
from .sparsegraph import load_from_npz
import random
import numba


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.shape[1]]

        return sp.csr_matrix((data, indices, indptr), shape=shape)


def split_random(seed, n, n_train, n_val):
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx


def get_data(dataset_path, seed, ntrain_div_classes, normalize_attr=None):
    '''
    Get data from a .npz-file.

    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    normalize_attr
        Normalization scheme for attributes. By default (and in the paper) no normalization is used.

    '''
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(g.attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(g.attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(g.attr_matrix):
            attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        attr_matrix = g.attr_matrix

    # helper that speeds up row indexing
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)
    else:
        attr_matrix = attr_matrix

    # split the data into train/val/test
    num_classes = g.labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)

    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx


def get_reddit(dataset_path, seed, ntrain_div_classes):
    dataset = Reddit(dataset_path)
    data = dataset[0]

    n, d = data.x.shape

    num_classes = data.y.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)

    attr_matrix = sp.csr_matrix(data['x'], dtype=np.float32)
    attr_matrix = SparseRowIndexer(attr_matrix)

    edges = data.edge_index.T
    adj_matrix = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(data.y.shape[0], data.y.shape[0]),
                        dtype=np.float32)
    labels = data.y.numpy()
    return adj_matrix, attr_matrix, labels, train_idx, val_idx, test_idx

def get_max_memory_bytes():
    return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def clustering_coefficient(adj_matrix, indx):
    coef = []
    for v in indx:
        neighbors = adj_matrix.indices[adj_matrix.indptr[v]:adj_matrix.indptr[v+1]]
        if v in neighbors:
            neighbors = np.delete(neighbors, np.argwhere(neighbors == v))
        l = len(neighbors)
        if l <= 1:
            coef.append(0.)
            continue

        links = 0.0
        if l <= 35:
            for i in range(l-1):
                for j in range(i, l):
                    w = neighbors[i]
                    u = neighbors[j]
                    if u in adj_matrix.indices[adj_matrix.indptr[w]:adj_matrix.indptr[w+1]]:
                        links += 1.
            coef.append(2.0*links/(l*(l-1)))
        else:
            for _ in range(1000):
                neighbors_list = list(neighbors)
                w, u = random.sample(neighbors_list, 2)
                if u in adj_matrix.indices[adj_matrix.indptr[w]:adj_matrix.indptr[w + 1]]:
                    links += 1.
            coef.append(links / 1000.)
    return coef
