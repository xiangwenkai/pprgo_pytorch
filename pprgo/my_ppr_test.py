import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from pprgo import my_ppr
import time
import scipy.sparse as sp



I = [0,1,0,1,2,3]
J = [1,2,3,0,1,0]
V = [1,1,1,1,1,1]
mat1 = sp.coo_matrix((V, (I, J)), shape=(10,10))
mat1 = mat1.tocsr()

I = [0,1,0,1,2,3,0,4]
J = [1,2,3,0,1,0,4,0]
V = [1,1,1,1,1,1,1,1]
mat2 = sp.coo_matrix((V, (I, J)), shape=(10,10))
mat2 = mat2.tocsr()


start = time.time()
ps, rs = my_ppr.DaynamicPPE(mat1, idx=[0,1,2,3,4], epsilon=1e-5, alpha=0.1)
ps, rs = my_ppr.calc_ppr(mat1, epsilon=1e-5, alpha=0.1, nodes=[0,1,2,3,4])
# 对孤立点初始化值为1，使之适合后续forwardpush运算
ps[4][4] = 0
rs[4][4] = 1

# 尝试forwardpush初始化，失败
# delta_edge = [(i,j,'i') for i, j in zip(I,J)]
# ps, rs = my_ppr.DynamicSNE(mat1, delta_edge=delta_edge, epsilon=1e-5, alpha=0.1, S=[0,1,2,3,4])

# n = len(mat2.indptr)
# adj_degree = [mat2.indptr[i + 1] - mat2.indptr[i] for i in range(n - 1)]
# u=4
# ps[u], rs[u] = my_ppr._calc_ppr_node(u, mat2, alpha=0.1, epsilon=1e-4, adj_degree=adj_degree,
#                                                         p_pre=ps[u], r_pre=rs[u])

ps_new, rs_new = my_ppr.DynamicSNE(mat2, delta_edge=[(4,0,'i')], epsilon=1e-4, alpha=0.1, S=[0,1,2,3,4], p_pre=ps, r_pre=rs)
print(f"t1: {time.time() - start}")


p1 = [[ps_new[i][j] for j in sorted(ps_new[i])] for i in ps_new.keys()]

start = time.time()
ps_G1, rs_G1 = my_ppr.DaynamicPPE(mat2, idx=[0,1,2,3,4], epsilon=1e-5, alpha=0.1)
print(f"t2: {time.time() - start}")


p2 = [[ps_G1[i][j] for j in sorted(ps_G1[i])] for i in ps_G1.keys()]
print(np.linalg.norm(p1 - np.asarray(p2), axis=1))


# ppr, _ = my_ppr.calc_ppr(G1, alpha=0.1, epsilon=1e-8, nodes=list(G0.nodes))

