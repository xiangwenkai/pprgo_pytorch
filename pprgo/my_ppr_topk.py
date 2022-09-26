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
def ForwardPushInit(p, r, s, indptr, indices, epsilon, alpha, adj_degree):
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
        for vnode in list(indices[indptr[snode]: indptr[snode+1]]):
            if vnode in r:
                r[vnode] += res
            else:
                r[vnode] = res
            res_vnode = r[vnode]
            if abs(res_vnode) >= epsilon * adj_degree[vnode]:
                if vnode not in enque:
                    enque.append(vnode)
        r[snode] = float32_0
    return p, r


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def ForwardPushUpdate(p, r, update_node, indptr, indices, epsilon, alpha, adj_degree,k=100):
    # while exist unode, abs(r[unode])>epsilon*adj_degree[unode]:
    enque = list(update_node)
    # enque = list(r.keys())
    while enque:
        unode = enque.pop()
        if abs(r[unode]) > epsilon*adj_degree[unode]:
            _val = alpha*r[unode]
            if unode in p:
                p[unode] += _val
            else:
                p[unode] = _val
            res = (1 - alpha) * r[unode] / adj_degree[unode]
            for vnode in list(indices[indptr[unode]: indptr[unode+1]]):
                if vnode in r:
                    r[vnode] += res
                else:
                    r[vnode] = res
                res_vnode = r[vnode]
                if abs(res_vnode) >= epsilon * adj_degree[vnode]:
                    if vnode not in enque:
                        enque.append(vnode)
            r[unode] = 0

    p_topk = sorted(p.items(), key=lambda x: x[1], reverse=True)[:k]
    r_topk = sorted(r.items(), key=lambda x: x[1], reverse=True)[:k]
    p = Dict.empty(key_type=numba.types.int64,
                      value_type=numba.types.float32)
    r = Dict.empty(key_type=numba.types.int64,
                      value_type=numba.types.float32)
    for x in p_topk:
        p[x[0]] = x[1]
    for x in r_topk:
        r[x[0]] = x[1]
    return p, r


@numba.njit(cache=True)
def DynamicSNE(indptr, indices, delta_edge, epsilon, alpha, adj_degree, p_pre=None, r_pre=None):
    update_node = []
    for change in delta_edge:
        u, v = change[0], change[1]

        if u not in p_pre:
            continue
        adj_degree[u] += 1
        delta_pu = p_pre[u] / (adj_degree[u] - 1)
        p_pre[u] += delta_pu
        r_pre[u] = r_pre.setdefault(u, 0) - delta_pu / alpha
        r_pre[v] = r_pre.setdefault(v, 0) + delta_pu / alpha - delta_pu

        for _ in [u, v]:
            if abs(r_pre[_]) > epsilon*adj_degree[_]:
                update_node.append(_)
    update_node = list(set(update_node))
    if len(update_node)>0:
        p_new, r_new = ForwardPushUpdate(p=p_pre, r=r_pre, update_node=update_node, indptr=indptr, indices=indices, epsilon=epsilon,
                                         alpha=alpha, adj_degree=adj_degree)
        return p_new, r_new
    return p_pre, r_pre


@numba.njit(cache=True, parallel=True)
def DynamicPPE_init(indptr,indices,adj_degree, S, epsilon, alpha):
    p = {}
    r = {}

    for s in S:
        p[s] = Dict.empty(key_type=numba.types.int64,
                        value_type=numba.types.float32)
        r[s] = Dict.empty(key_type=numba.types.int64,
                        value_type=numba.types.float32)
        p[s][s] =numba.float32(0)
        r[s][s] = numba.float32(1)
    for s in S:
        p[s], r[s] = ForwardPushInit(p=p[s], r=r[s], s=s, indptr=indptr, indices=indices, epsilon=epsilon,
                                         alpha=alpha, adj_degree=adj_degree)
    return p, r


@numba.njit(cache=True, parallel=True)
def DynamicPPE_update(indptr,indices,adj_degree, delta_edge, S, epsilon, alpha, p_pre, r_pre):
    for s in S:
        p_pre[s], r_pre[s] = DynamicSNE(indptr, indices, delta_edge=delta_edge,
                                epsilon=epsilon, alpha=alpha,
                                adj_degree=adj_degree, p_pre=p_pre[s], r_pre=r_pre[s])
    return p_pre, r_pre





