import numpy as np
import torch
import copy
import scipy.sparse as sp
from LDS_GNN_modify.lds_gnn.main_graph import train_GCN

labels = np.load('../../labels.npy')
# embeddings = np.load('hidden.npy')
adj = np.load('../../adj.npy')
features = np.load('../../features.npy')

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    # print(n_edges, edges)
    print(edges.shape)
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        print('remove edges:',n_remove)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        print('add edges:',n_add)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

def cal_inter_par(adj, labels):
    inter = 0
    intra = 0
    labels = np.argmax(labels, 1)
    row, col = np.where(adj != 0.)
    for i, j in zip(row, col):
        if labels[i] != labels[j]:
            inter += 1
        else:
            intra += 1
    return inter, intra, inter/len(row), intra/len(row), len(row)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

norm_adj = normalize_adj(adj).todense()
norm_adj = norm_adj / np.sum(norm_adj, 1)
# print(norm_adj, np.sum(norm_adj, 1))
norm_adj_2 = np.dot(norm_adj, norm_adj)
print(norm_adj_2)
labels = np.argmax(labels, 1)
# idx = {'train':range(140), 'val':range(500,1000), 'test':range(adj.shape[0]-1000, adj.shape[0])}
# train_GCN(norm_adj, idx, labels, features)
# row, col = np.where(norm_adj_2 != 0)
# for i,j in zip(row,col):
#     if j < i:
#         continue
#     else:
#         if labels[i] != labels[j]:
#             print(norm_adj_2[i,j])
adj_pred = sample_graph_det(norm_adj_2, norm_adj_2, 10, False)
print(adj_pred)