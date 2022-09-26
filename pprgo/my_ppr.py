import time

import numpy as np
import numba
import scipy.sparse as sp
import sklearn
from numba.typed import Dict
# from pprgo.utils import SparseRowIndexer

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
    return p, r


def _calc_ppr_node(inode, adj, alpha, epsilon, adj_degree, p_pre=None, r_pre=None):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    if p_pre is None:
        p = {inode: f32_0}
    else:
        p = p_pre
    if r_pre is None:
        r = {}
        r[inode] = alpha
    else:
        r = r_pre
        r[inode] = alpha
    q = [inode]

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in list(adj.indices[adj.indptr[unode]: adj.indptr[unode+1]]):
            _val = (1 - alpha) * res / adj_degree[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * adj_degree[vnode]:
                if vnode not in q:
                    q.append(vnode)
    return p, r


def calc_ppr(adj, alpha, epsilon, nodes, p_pre=None, r_pre=None):
    n = len(adj.indptr)
    adj_degree = [adj.indptr[i + 1] - adj.indptr[i] for i in range(n - 1)]

    ps = {}
    rs = {}
    for i, node in enumerate(nodes):
        if p_pre is not None and node in p_pre:
            p, r = _calc_ppr_node(node, adj, alpha, epsilon, adj_degree=adj_degree,
                                  p_pre=p_pre[node], r_pre=r_pre[node])
        else:
            p, r = _calc_ppr_node(node, adj, alpha, epsilon, adj_degree=adj_degree)
        ps[node] = p
        rs[node] = r
        # js.append(list(p.keys()))
        # vals.append(list(p.values()))
    return ps, rs


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def ForwardPush(p, r, s, indptr, indices, epsilon, alpha, adj_degree):
    # count = numba.int16(0)
    float32_0 = numba.float32(0)
    # 如果孤立，则不必更新
    if indptr[s] == indptr[s+1]:
        return p, r
    enque = [s]
    while len(enque) > 0:
        snode = enque.pop()
        if snode not in r:
            continue
        _val = alpha*r[snode]
        if snode in p:
            p[snode] += _val
        else:
            p[snode] = _val
        res = (1 - alpha) * r[snode] / adj_degree[snode]
        # count += 1
        for vnode in list(indices[indptr[snode]: indptr[snode+1]]):
            if vnode in r:
                r[vnode] += res
            else:
                r[vnode] = res
            res_vnode = r[vnode]
            if abs(res_vnode) >= epsilon * adj_degree[vnode]:
                if vnode not in enque:
                    enque.append(vnode)
            # count += 1
        r[snode] = float32_0
    return p, r

@numba.njit(cache=True, parallel=True)
def DynamicSNE(indptr, indices, delta_edge, epsilon, alpha, s, adj_degree, p_pre=None, r_pre=None):
    # d_count = 0
    for change in delta_edge:
        u, v = change[0], change[1]

        delta_pu = p_pre[u] / max(adj_degree[u] - 1, 1)
        p_pre[u] += delta_pu
        r_pre[u] = r_pre[u] - delta_pu / alpha
        r_pre[v] = r_pre[v] + delta_pu / alpha - delta_pu

        delta_pv = p_pre[v] / max(adj_degree[v] - 1, 1)
        p_pre[v] += delta_pv
        r_pre[v] = r_pre[v] - delta_pv / alpha
        r_pre[u] = r_pre[u] + delta_pv / alpha - delta_pv
        '''
        u, v, op = change[0], change[1], change[2]
        if op == 'i':
            if u in p_pre and adj_degree[u] > 1:
                delta_pu = p_pre[u] / (adj_degree[u] - 1)
                p_pre[u] += delta_pu
                r_pre[u] = r_pre[u] - delta_pu / alpha
                r_pre[v] = r_pre[v] + delta_pu / alpha - delta_pu if v in r_pre else delta_pu / alpha - delta_pu
                # d_count+=1
            if v in p_pre and adj_degree[v] > 1:
                delta_pv = p_pre[v] / (adj_degree[v] - 1)
                p_pre[v] += delta_pv
                r_pre[v] = r_pre[v] - delta_pv / alpha
                r_pre[u] = r_pre[u] + delta_pv / alpha - delta_pv if u in r_pre else delta_pv / alpha - delta_pv
                # d_count += 1
        elif op == 'd':
            if u in p_pre:
                delta_pu = -p_pre[u] / (adj_degree[u] + 1)
                p_pre[u] += delta_pu
                r_pre[u] = r_pre[u] - delta_pu / alpha
                r_pre[v] = r_pre[v] + delta_pu / alpha - delta_pu if v in r_pre else delta_pu / alpha - delta_pu
                # d_count += 1
            if v in p_pre:
                delta_pv = -p_pre[v] / (adj_degree[v] + 1)
                p_pre[v] += delta_pv
                r_pre[v] = r_pre[v] - delta_pv / alpha
                r_pre[u] = r_pre[u] + delta_pv / alpha - delta_pv if u in r_pre else delta_pv / alpha - delta_pv
                # d_count += 1
    '''
    p_new, r_new = ForwardPush(p=p_pre, r=r_pre, s=s, indptr=indptr, indices=indices, epsilon=epsilon,
                                     alpha=alpha, adj_degree=adj_degree)
    return p_new, r_new

@numba.njit(cache=True, parallel=True)
def DynamicPPE_init(indptr,indices,adj_degree, S, epsilon, alpha):
    p = {}
    r = {}

    for s in S:
        p[s] = Dict.empty(key_type=numba.types.int32,
                        value_type=numba.types.float32)
        r[s] = Dict.empty(key_type=numba.types.int32,
                        value_type=numba.types.float32)
        p[s][s] =numba.float32(0)
        r[s][s] = numba.float32(1)
    # count = 0
    for s in S:
        p[s], r[s] = ForwardPush(p=p[s], r=r[s], s=s, indptr=indptr, indices=indices, epsilon=epsilon,
                                         alpha=alpha, adj_degree=adj_degree)
        # count += d
        # p, r = DynamicSNE(indptr, indices, delta_edge=delta_edge,
        #                         epsilon=epsilon, alpha=alpha, s=s,
        #                         adj_degree=adj_degree, p_pre=p, r_pre=r)
    return p, r

@numba.njit(cache=True, parallel=True)
def DynamicPPE_update(indptr,indices,adj_degree, delta_edge, S, epsilon, alpha, p_pre, r_pre):
    # f_count, d_count = 0, 0
    for s in S:
        # t=time.time()
        # p_pre[s], r_pre[s] = ForwardPush(p=p_pre[s], r=r_pre[s], s=s, indptr=indptr, indices=indices, epsilon=epsilon,
        #                            alpha=alpha, adj_degree=adj_degree)
        # print(time.time()-t)
        p_pre[s], r_pre[s] = DynamicSNE(indptr, indices, delta_edge=delta_edge,
                                epsilon=epsilon, alpha=alpha, s=s,
                                adj_degree=adj_degree, p_pre=p_pre[s], r_pre=r_pre[s])

        # f_count += f_
        # d_count += d_
    return p_pre, r_pre


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def ppr_convex(adj_matrix, ps, idx, normalization='col'):
    nnodes = len(idx)
    neighbors, weights = [np.zeros(0, dtype=np.int64)] * nnodes, [np.zeros(0, dtype=np.float32)] * nnodes
    for i in numba.prange(len(idx)):
        # val = np.array(list(ps[i].values()))
        # row = val.nonzero()[0]
        # val = val[row]
        # kl = np.array(list(ps[i].keys()))[row]
        val = np.array(list(ps[idx[i]].values()))
        kl = np.array(list(ps[idx[i]].keys()))
        # idx_val = np.argsort(val)
        # neighbors[i] = kl[idx_val]
        # weights[i] = val[idx_val]
        neighbors[i] = kl
        weights[i] = val
    n_all = adj_matrix.shape[0]
    ppr_matrix = construct_sparse(neighbors, weights, (nnodes, n_all)).tocsr()

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = ppr_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        ppr_matrix.data = deg_sqrt[idx[row]] * ppr_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = ppr_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        ppr_matrix.data = deg[idx[row]] * ppr_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")
    return ppr_matrix

def attr_mat_convex(df_attr, normalize_attr):
    '''
    Parameters
    ----------
    df_attr:第一列为'node'，其余列为特征
    normalize_attr
    Returns
    -------
    '''
    features = list(df_attr.columns)
    features.remove('node')
    nnodes = df_attr.shape[0]
    neighbors = [0] * nnodes
    l = list(np.arange(df_attr.shape[1] - 1))
    for i in df_attr['node']:
        neighbors[i] = l
    weights = df_attr[features].values.tolist()
    attr_matrix = construct_sparse(neighbors, weights, (nnodes, df_attr.shape[1] - 1)).tocsr()

    if sp.isspmatrix(attr_matrix):
        attr_matrix = attr_matrix.tocsr().astype(np.float32)
    elif isinstance(attr_matrix, np.ndarray):
        attr_matrix = attr_matrix.astype(np.float32)
    else:
        raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                         .format(type(attr_matrix)))

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(attr_matrix):
            attr_norms = sp.linalg.norm(attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        pass
    return attr_matrix

def split_random(seed, n, n_train, n_val):
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx


def load_data(loader):
    import networkx as nx
    nnodes = loader['adj_matrix.indptr'].size
    indices = loader['adj_matrix.indices']
    indptr = loader['adj_matrix.indptr']
    for i in range(nnodes):
        indices_co = [0]*loader['adj_matrix.indices'].size
        indices_co[indptr[i]:indptr[i + 1]] = i
    G = nx.Graph().add_edges_from(list(tuple(zip(indices_co, indices))))





