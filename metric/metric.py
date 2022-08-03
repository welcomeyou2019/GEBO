import numpy as np
# from gcn.utils import load_data
import pickle
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from copy import copy
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import pandas as pd


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):

    if dataset_str == 'pubmed':
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../lds_gnn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../lds_gnn/data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)

        if not isinstance(features, sp.csr.csr_matrix):
            features = sp.csr_matrix(features)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        nclass = np.max(np.argmax(labels, 1)) + 1

    elif dataset_str == 'terrorist':
        raw_data = pd.read_csv('../lds_gnn/data/terrorist-attacks/terrorist_attack.nodes', sep='\t', header=None)
        num = raw_data.shape[0]

        a = list(raw_data.index)
        b = list(raw_data[0])
        c = zip(b, a)
        map = dict(c)

        features = raw_data.iloc[:, 1:-1]
        features = sp.csr_matrix(features)

        labels = pd.get_dummies(raw_data[107])
        labels = np.array(labels)
        d = np.argmax(labels, 1)

        lab_dic = {i: [] for i in range(6)}
        for ind, i in enumerate(d):
            lab_dic[i].append(ind)

        idx_train = []
        idx_val = []
        idx_test = []
        for i in lab_dic:
            idx_train.extend(lab_dic[i][:int(0.1 * len(lab_dic[i]))])
            idx_val.extend(lab_dic[i][int(0.1 * len(lab_dic[i])):int(0.3 * len(lab_dic[i]))])
            idx_test.extend(lab_dic[i][int(0.3 * len(lab_dic[i])):])

        raw_data_cites = pd.read_csv('../lds_gnn/data/terrorist-attacks/terrorist_attack_loc.edges', sep='\\s+', header=None)
        raw_data_cites_1 = pd.read_csv('../lds_gnn/data/terrorist-attacks/terrorist_attack_loc_org.edges', sep='\\s+', header=None)
        adj = np.zeros((num, num))
        # 创建邻接矩阵
        for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
            x = map[i]
            y = map[j]  # 替换论文编号为[0,2707]
            adj[x][y] = adj[y][x] = 1  # 有引用关系的样本点之间取1
        for i, j in zip(raw_data_cites_1[0], raw_data_cites_1[1]):
            x = map[i]
            y = map[j]  # 替换论文编号为[0,2707]
            adj[x][y] = adj[y][x] = 1  # 有引用关系的样本点之间取1
        adj = sp.coo_matrix(adj)
        adj.setdiag(1)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        nclass = labels.shape[1]


    else:
        tvt_nids = pickle.load(open(f'../lds_gnn/data/graphs/{dataset_str}_tvt_nids.pkl', 'rb'))
        adj = pickle.load(open(f'../lds_gnn/data/graphs/{dataset_str}_adj.pkl', 'rb'))
        features = pickle.load(open(f'../lds_gnn/data/graphs/{dataset_str}_features.pkl', 'rb'))
        labels = pickle.load(open(f'../lds_gnn/data/graphs/{dataset_str}_labels.pkl', 'rb'))
        if not isinstance(features, sp.csr.csr_matrix):
            features = sp.csr_matrix(features)
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if len(labels.shape) == 1:
            nclass = np.max(labels) + 1
            labels = np.eye(nclass)[labels]
        else:
            nclass = labels.shape[-1]
        idx_train = tvt_nids[0]
        idx_val = tvt_nids[1]
        idx_test = tvt_nids[2]
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask] = labels[train_mask]
        y_val[val_mask] = labels[val_mask]
        y_test[test_mask] = labels[test_mask]
        # print(y_train.shape, y_val.shape)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, nclass, idx_val, labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized

def modularity(A, labels):
    m = np.sum(A) / 2
    K = np.sum(A, 1)
    delta = np.zeros(A.shape)
    minus = np.zeros(A.shape)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if labels[i] == labels[j]:
                delta[i,j] = 1
                minus[i,j] = A[i,j] - K[i] * K[j] / (2*m)
    Q = np.sum(minus * delta) / (2*m)
    return Q
# name = 'cora'
for name in ['cora']:#, 'citeseer', 'airport', 'terrorist', 'ppi']:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, nclass, idx_val, labels = load_data(name)
    adj = adj.todense()
    labels = np.argmax(labels, 1)
    # print(labels)
    process = preprocess_adj(adj).todense()
    pred = np.load('z0_' + name + '.npy')
    pred_1 = np.load('z1_' + name + '.npy')
    # print(pred)

    intra_org = 0
    intra_pred = 0
    intra_pred_1 = 0
    count = 0
    # for i in range(adj.shape[0]):
    #     row,col = np.where(process[i]!=0.)
    #     for j in col:
    #         if labels[i] == labels[j]:
    #             count +=1
    #             intra_org += process[i,j]
    #             intra_pred += pred[i, j]
    #             intra_pred_1 += pred_1[i, j]
    # I_org = intra_org / count
    # I_pred = intra_pred / count
    # I_pred_1 = intra_pred_1 / count
    # print(I_org, I_pred, I_pred_1)
    # Q1 = modularity(process, labels)
    # Q2 = modularity(pred, labels)
    # Q3 = modularity(pred_1, labels)
    # print(Q1, Q2, Q3)
    intra_count = 0
    renew_intra = 0
    adj_renew = np.zeros(adj.shape)
    # print(process, pred, pred_1)
    delta_1 = pred - process
    delta_2 = pred_1 - process
    avg_delta = (pred+pred_1)/2 - process
    delta_1_c = 0
    delta_1_inter = 0
    delta_2_c = 0
    delta_2_inter = 0
    avg_del = 0
    avg_del_inter = 0
    # print(delta_1, delta_2, avg_delta)
    for i in range(adj.shape[0]):
        row, col = np.where(process[i]!=0)
        for j in col:
            if labels[i] == labels[j]:
                intra_count += 1
            if pred[i,j] > process[i,j] and pred_1[i,j] > process[i,j]:
                if pred[j,i] > process[j,i] and pred_1[j,i] > process[j,i]:
                    adj_renew[i,j] = 1
                    adj_renew[j,i] = 1
                    if labels[i] == labels[j]:
                        renew_intra += 1
            if delta_1[i,j] < -1e-2:
                delta_1_c += 1
                if labels[i] != labels[j]:
                    delta_1_inter +=1
                    # print(delta_1[i,j])
            if delta_2[i,j] < -1e-2:
                delta_2_c += 1
                if labels[i] != labels[j]:
                    delta_2_inter +=1
            if avg_delta[i,j] < -1e-2:
                avg_del +=1
                if labels[i] != labels[j]:
                    avg_del_inter +=1
    # print(intra_count, np.sum(adj) - intra_count, intra_count/np.sum(adj), np.sum(adj))
    # print(renew_intra, np.sum(adj_renew) - renew_intra, renew_intra/np.sum(adj_renew), np.sum(adj_renew))
    print(delta_1_c, delta_1_inter, delta_1_inter/delta_1_c, delta_2_c, delta_2_inter,delta_2_inter/delta_2_c, avg_del, avg_del_inter, avg_del_inter/avg_del)