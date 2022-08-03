

"""This module contains methods to load and manage datasets. For graph based data, it mostly resorts to gcn package"""
from scipy import sparse
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
try:
    from lds_gnn.utils import Config, upper_triangular_mask
except ImportError as e:
    from utils import Config, upper_triangular_mask


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        # self.f1 = 'load_data_del_edges'
        self.f1 = 'load_data'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        # res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        res = load_data(dataset_str=self.dataset_name)
        if self.f2:
            print('f2',**self.kwargs_f2)
            res = reorganize_data_for_es(res, **self.kwargs_f2, seed=self.seed, dataset_name=self.dataset_name)
        return res


class EdgeDelConfigData(ConfigData):
    def __init__(self, **kwargs):
        self.prob_del = 0
        self.enforce_connected = False
        super().__init__(**kwargs)
        self.kwargs_f1['prob_del'] = self.prob_del
        if not self.enforce_connected:
            self.kwargs_f1['enforce_connected'] = self.enforce_connected
        del self.prob_del
        del self.enforce_connected


class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.n_es = None
        self.scale = None
        super().__init__(**kwargs)

    def load(self):
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
        elif self.dataset_name == 'fma':
            import os
            data = np.load('%s/fma/fma.npz' % os.getcwd())
        elif self.dataset_name == '20news10':
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            categories = ['alt.atheism',
                          'comp.sys.ibm.pc.hardware',
                          'misc.forsale',
                          'rec.autos',
                          'rec.sport.hockey',
                          'sci.crypt',
                          'sci.electronics',
                          'sci.med',
                          'sci.space',
                          'talk.politics.guns']
            data = fetch_20newsgroups(subset='all', categories=categories)
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
        else:
            raise AttributeError('dataset not available')

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if self.scale:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val + self.n_es,
                                                        test_size=n - self.n_train - self.n_val - self.n_es,
                                                        stratify=y)
        train, es, y_train, y_es = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train + self.n_val, test_size=self.n_es,
                                                    stratify=y_train)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                      train_size=self.n_train, test_size=self.n_val,
                                                      stratify=y_train)

        train_mask = np.zeros([n, ], dtype=bool)
        train_mask[train] = True
        val_mask = np.zeros([n, ], dtype=bool)
        val_mask[val] = True
        es_mask = np.zeros([n, ], dtype=bool)
        es_mask[es] = True
        test_mask = np.zeros([n, ], dtype=bool)
        test_mask[test] = True

        return np.zeros([n, n]), np.zeros([n, n]), features, ys, train_mask, val_mask, es_mask, test_mask


def graph_delete_connections(prob_del, seed, adj, features, y_train,
                             *other_splittables, to_dense=False,
                             enforce_connected=False, dataset_name='cora'):
    rnd = np.random.RandomState(seed)
    if dataset_name == 'cora' or dataset_name == 'citeseer':
        features = preprocess_features(features)

    if to_dense:
        features = features.toarray()
        adj = adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * upper_triangular_mask(
        adj.shape, as_array=True)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)

    return (adj, del_adj, features, y_train) + other_splittables

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):

    if dataset_str == 'pubmed':
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
        raw_data = pd.read_csv('data/terrorist-attacks/terrorist_attack.nodes', sep='\t', header=None)
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

        raw_data_cites = pd.read_csv('data/terrorist-attacks/terrorist_attack_loc.edges', sep='\\s+', header=None)
        raw_data_cites_1 = pd.read_csv('data/terrorist-attacks/terrorist_attack_loc_org.edges', sep='\\s+', header=None)
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
        tvt_nids = pickle.load(open(f'data/graphs/{dataset_str}_tvt_nids.pkl', 'rb'))
        adj = pickle.load(open(f'data/graphs/{dataset_str}_adj.pkl', 'rb'))
        features = pickle.load(open(f'data/graphs/{dataset_str}_features.pkl', 'rb'))
        labels = pickle.load(open(f'data/graphs/{dataset_str}_labels.pkl', 'rb'))
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
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, nclass, idx_val, labels, idx_train, idx_val, idx_test


# def load_data_del_edges(prob_del=0, seed=0, to_dense=False, enforce_connected=False,
#                         dataset_name='cora'):
    # if dataset_name == 'cora' or dataset_name == 'citeseer':
    # res = graph_delete_connections(prob_del, seed, *load_data(dataset_name), to_dense=to_dense,
    #                                enforce_connected=enforce_connected)
    # res = load_data(dataset_name)
    # else:
    #     tvt_nids = pickle.load(open(f'data/graphs/{dataset_name}_tvt_nids.pkl', 'rb'))
    #     idx_train = tvt_nids[0]
    #     idx_val = tvt_nids[1]
    #     idx_test = tvt_nids[2]
    #
    #     adj = pickle.load(open(f'data/graphs/{dataset_name}_adj.pkl', 'rb'))
    #     adj = adj - np.eye(adj.shape[0])
    #     features = pickle.load(open(f'data/graphs/{dataset_name}_features.pkl', 'rb'))
    #     adj = sparse.csr_matrix(adj)
    #     features = sparse.csr_matrix(features)
    #     labels = pickle.load(open(f'data/graphs/{dataset_name}_labels.pkl', 'rb'))
    #     # print(labels)
    #
    #     if len(labels.shape) == 1:
    #         one_hot_labels = np.zeros((adj.shape[0], max(labels) + 1))
    #         for ind,i in enumerate(labels):
    #             one_hot_labels[ind, i] = 1
    #         labels = copy(one_hot_labels)
    #     # print(labels)
    #     # print(labels.shape)
    #     train_mask = sample_mask(idx_train, labels.shape[0])
    #     val_mask = sample_mask(idx_val, labels.shape[0])
    #     test_mask = sample_mask(idx_test, labels.shape[0])
    #     # print(np.sum(train_mask),np.sum(val_mask))
    #
    #     y_train = np.zeros(labels.shape)
    #     y_val = np.zeros(labels.shape)
    #     y_test = np.zeros(labels.shape)
    #     y_train[train_mask, :] = labels[train_mask, :]
    #     y_val[val_mask, :] = labels[val_mask, :]
    #     y_test[test_mask, :] = labels[test_mask, :]
    #     # print(y_train)
    #
    #     # y_train = labels[train_mask]
    #     # y_val = labels[val_mask]
    #     # y_test = labels[test_mask]
    #     res = graph_delete_connections(prob_del, seed, *(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask), to_dense=to_dense,
    #                                    enforce_connected=enforce_connected, dataset_name=dataset_name)
    # return res


def reorganize_data_for_es(loaded_data, seed=0, es_n_data_prop=0.5, dataset_name='cora'):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, nclass, idx_val, labels, idx_train, idx_val, idx_test = loaded_data
    ys = y_train + y_val + y_test #2708,7
    if dataset_name == 'cora' or dataset_name == 'citeseer':
        features = preprocess_features(features)
    else:
        features = sparse_to_tuple(features)
    # msk1, msk2 = divide_mask(es_n_data_prop, np.sum(val_mask), seed=seed) #500
    idx_val_sep = np.random.choice(idx_val, int(es_n_data_prop * len(idx_val)), replace=False)
    idx_es = np.array(list(set(idx_val) - set(idx_val_sep)))
    # mask_val = np.array(val_mask)
    # mask_es = np.array(val_mask)

    mask_val = sample_mask(idx_val_sep, labels.shape[0])
    mask_es = sample_mask(idx_es, labels.shape[0])

    y_val_sep = np.zeros(labels.shape)
    y_es = np.zeros(labels.shape)

    y_val_sep[mask_val] = labels[mask_val]
    y_es[mask_es] = labels[mask_es]
    # print(y_val_sep.shape, np.sum(y_val_sep))

    # mask_val[mask_val] = msk2
    # mask_es[mask_es] = msk1
    # y_val_1 = ys[mask_val]
    # y_es = ys[mask_es]

    return adj, y_train, y_val, y_val_sep, y_es, y_test, features, train_mask, mask_val, mask_es, test_mask, nclass, idx_train, idx_val, idx_test


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return features
    return sparse_to_tuple(features)

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

def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs
