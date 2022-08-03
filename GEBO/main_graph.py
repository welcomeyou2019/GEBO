import networkx as nx
from LDS_GNN_modify.lds_gnn.analysis_adj import load_data, reorganize_data_for_es, train_GCN
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from lds_gnn.models import gcn_train
from lds_gnn.gcn.metrics import masked_accuracy, masked_softmax_cross_entropy
import scipy
import torch
from tqdm import tqdm

def cal_subgraph(adj):
    G = nx.from_numpy_array(adj)
    subgraph = nx.connected_components(G)
    return subgraph

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_data(dataset_name):
    res = load_data(dataset_str=dataset_name)
    adj, y_train, y_val, y_val_sep, y_es, y_test, features, train_mask, val_mask, es_mask, test_mask, \
        nclass, idx_train, idx_val, idx_test, labels= reorganize_data_for_es(res, seed=1979, dataset_name=dataset_name)
    features = scipy.sparse.coo_matrix((np.array(features[1]), (features[0].T[0], features[0].T[1])), shape=features[2]).todense()
    np.save(dataset_name +'_features.npy', features)
    adj = adj.todense()
    np.save(dataset_name+'_adj.npy', adj)
    np.save(dataset_name+'_labels.npy', labels)
    return idx_train, idx_val, idx_test


def get_main_graph(dataset_name, idx_test):
    adj = np.load(dataset_name+'_adj.npy')
    labels = np.load(dataset_name+'_labels.npy')
    features = np.load(dataset_name+'_features.npy')

    labels_hot = np.argmax(labels, 1)
    nclass = max(np.argmax(labels, 1)) + 1
    graphs = list(cal_subgraph(adj))

    len_graphs = [len(i) for i in graphs]
    # print(len_graphs)
    main_graph = list(graphs[len_graphs.index(max(len_graphs))])

    test_index = [i for i in main_graph if i in idx_test]
    # print(idx_test)
    # print(main_graph)
    # print(test_index)

    rest_graph = [i for i in range(features.shape[0]) if i not in main_graph]

    main_array = np.array(main_graph)
    rest_array = np.array(rest_graph)
    # print(np.where(main_array >= adj.shape[0] - 1000)[0][0])

    graph_mask = sample_mask(main_graph, labels.shape[0])
    rest_mask = sample_mask(rest_graph, labels.shape[0])

    main_adj = adj[graph_mask]
    main_adj = main_adj[:, graph_mask]

    rest_adj = adj[rest_mask]
    rest_adj = rest_adj[:, rest_mask]

    main_features = features[graph_mask]
    rest_features = features[rest_mask]
    # np.save('main_features.npy', main_features)

    main_labels = labels[graph_mask]
    rest_labels = labels[rest_mask]
    idx = {'train':[], 'val':[], 'test':[]}

    label_idx = {i:[] for i in range(nclass)}
    for i in range(main_adj.shape[0]):
        if len(label_idx[labels_hot[i]]) < 20:
            label_idx[labels_hot[i]].append(i)
    for i in range(nclass):
        idx['train'].extend(label_idx[i])
    idx['val'] = range(500, 1000)
    # print(np.where(main_array >= (adj.shape[0] - 1000))[0])
    # idx['test'] = range(np.where(main_array >= (adj.shape[0] - 1000))[0][0], main_adj.shape[0]) #915

    idx['test'] = main_array[np.where(main_array >= (adj.shape[0] - 1000))[0]]
    return main_adj, main_labels, main_features, idx, rest_adj, rest_labels, rest_features, graphs, labels, features, adj

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
    return sparse_to_tuple(adj_normalized)

def setup_tf(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    if tf.get_default_session(): tf.get_default_session().close()
    return tf.InteractiveSession()

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

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def train_GCN(weighted_adj, idx, labels, features, seed=1979):
    ss = setup_tf(seed)
    nclass = max(np.argmax(labels,1)) + 1
    idx_train = idx['train']
    idx_val = idx['val']
    idx_test = idx['test']
    train_mask = sample_mask(idx_train, weighted_adj.shape[0])
    val_mask = sample_mask(idx_val, weighted_adj.shape[0])
    test_mask = sample_mask(idx_test, weighted_adj.shape[0])
    features = sparse_to_tuple(sp.coo_matrix(features))

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'nclass': nclass,
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    }

    # support = [sparse_to_tuple(sp.coo_matrix(weighted_adj))]
    support = [preprocess_adj(adj)]
    tr_fd = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    tr_fd.update({placeholders['dropout']: 0.5})

    test_fd = construct_feed_dict(features, support, y_test, test_mask, placeholders)

    val_all_fd = construct_feed_dict(features, support, y_val, val_mask, placeholders)
    feature_dim = features[2][1]
    out, ws, rep = gcn_train(placeholders, f_dim=feature_dim, num_layer=2)
    error = tf.identity(
        masked_softmax_cross_entropy(out, placeholders['labels'], placeholders['labels_mask']), 'error')
    tr_error = error + 5e-4 * tf.nn.l2_loss(ws[0])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    acc = tf.identity(masked_accuracy(out, placeholders['labels'], placeholders['labels_mask']), 'accuracy')

    opt_op = optimizer.minimize(tr_error)

    best_val_acc = 0
    best_test_acc = 0
    best_test_loss = 0
    best_out = 0

    ss.run(tf.global_variables_initializer())
    for epoch in range(100):
        out_tr = ss.run([opt_op, out, ws], tr_fd)
        loss_tr = ss.run(tr_error, tr_fd)
        acc_tr = ss.run(acc, {**tr_fd, out: out_tr[1]})

        out_val = ss.run(out, val_all_fd)
        loss_val = ss.run(tr_error, val_all_fd)
        acc_val = ss.run(acc, {**val_all_fd, out:out_val})

        out_test = ss.run(out, test_fd)
        loss_test = ss.run(tr_error, test_fd)
        acc_test = ss.run(acc, {**test_fd, out:out_test})

        # if acc_val > best_val_acc:
        #     best_val_acc = acc_val
        #     W = out_tr[-1]
        #     best_test_acc = acc_test
        #     best_test_loss = loss_test
        #     best_out = out_test

    print(acc_test, acc_test * len(idx['test']), len(idx['test']))
    return out_tr[-1], out_test

def get_rest_embeddings(rest_adj, rest_features, rest_labels, W):
    rest_adj = normalize_adj(rest_adj).todense()
    H1 = np.matmul(rest_adj, rest_features)
    H1 = np.matmul(H1, W[0])
    H1 = np.maximum(0, H1) #relu
    H2 = np.matmul(rest_adj, H1)
    H2 = np.matmul(H2, W[1])
    return H2

# class DAEGCTrainer():
#     def __init__(self, args):
#         self.num_clusters = args.num_clusters
#         self.max_epoch = args.max_epoch
#         self.lr = args.lr
#         self.weight_decay = args.weight_decay
#         self.T = args.T
#         self.gamma = args.gamma
#         self.device = args.device_id[0] if not args.cpu else "cpu"
#
#     def fit(self, model, data):
#         # edge_index_2hop = model.get_2hop(data.edge_index)
#         data.add_remaining_self_loops()
#         data.adj_mx = torch.sparse_coo_tensor(
#             torch.stack(data.edge_index),
#             torch.ones(data.edge_index[0].shape[0]),
#             torch.Size([data.x.shape[0], data.x.shape[0]]),
#         ).to_dense()
#         data = data.to(self.device)
#         edge_index_2hop = data.edge_index
#         model = model.to(self.device)
#         self.num_nodes = data.x.shape[0]
#
#         print("Training initial embedding...")
#         epoch_iter = tqdm(range(self.max_epoch))
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         with data.local_graph():
#             data.edge_index = edge_index_2hop
#             for epoch in epoch_iter:
#                 model.train()
#                 optimizer.zero_grad()
#                 z = model(data)
#                 loss = model.recon_loss(z, data.adj_mx)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
#
#         print("Getting cluster centers...")
#         kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(model(data).detach().cpu().numpy())
#         model.cluster_center = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, device=self.device))
#
#         print("Self-optimizing...")
#         epoch_iter = tqdm(range(self.max_epoch))
#         # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=self.weight_decay)
#         for epoch in epoch_iter:
#             self.cluster_center = model.cluster_center
#             model.train()
#             optimizer.zero_grad()
#             z = model(data)
#             Q = self.getQ(z)
#             if epoch % self.T == 0:
#                 P = self.getP(Q).detach()
#             loss = model.recon_loss(z, data.adj_mx) + self.gamma * self.cluster_loss(P, Q)
#             loss.backward()
#             optimizer.step()
#             epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
#         return model
#
#     def getQ(self, z):
#         Q = None
#         for i in range(z.shape[0]):
#             dis = torch.sum((z[i].repeat(self.num_clusters, 1) - self.cluster_center) ** 2, dim=1)
#             t = 1 / (1 + dis)
#             t = t / torch.sum(t)
#             if Q is None:
#                 Q = t.clone().unsqueeze(0)
#             else:
#                 Q = torch.cat((Q, t.unsqueeze(0)), 0)
#         # print("Q=", Q)
#         return Q
#
#     def getP(self, Q):
#         P = torch.sum(Q, dim=0).repeat(Q.shape[0], 1)
#         P = Q ** 2 / P
#         P = P / (torch.ones(1, self.num_clusters, device=self.device) * torch.sum(P, dim=1).unsqueeze(-1))
#         # print("P=", P)
#         return P
#
#     def cluster_loss(self, P, Q):
#         # return nn.MSELoss(reduce=True, size_average=False)(P, Q)
#         return torch.nn.KLDivLoss(reduce=True, size_average=False)(P.log(), Q)
dataset_name = 'airport'
idx_train, idx_val, idx_test = get_data(dataset_name)
# print(len(idx_train),len(idx_val),len(idx_test))
# print(idx_test)
main_adj, main_labels, main_features, idx, rest_adj, rest_labels, rest_features, graphs, labels, features, adj = get_main_graph(dataset_name, idx_test)
nclass = max(np.argmax(labels, 1)) + 1

idx_all = {}
idx_all['train'] = idx_train
idx_all['val'] = idx_val
idx_all['test'] = idx_test
_, emb = train_GCN(adj, idx_all, labels, features)
pred_all = np.argmax(emb, 1)
# main_graph = np.array(list(graphs[0]))
# print(idx['test'])
main_pred = pred_all[idx['test']]
main_labelssss = labels[idx['test']]

rest_idx = np.array([i for i in idx_all['test'] if i not in idx['test']])
# print(idx['test'])
rest_labelssss = labels[rest_idx]
rest_pred = pred_all[rest_idx]
# print(rest_pred.shape, rest_labelssss.shape)

print(np.sum(main_pred == np.argmax(main_labelssss,1)) / len(np.argmax(main_labelssss,1)), np.sum(main_pred == np.argmax(main_labelssss,1)), len(np.argmax(main_labelssss,1)))
print(np.sum(rest_pred == np.argmax(rest_labelssss,1)) / len(np.argmax(rest_labelssss,1)), np.sum(rest_pred == np.argmax(rest_labelssss,1)), len(np.argmax(rest_labelssss,1)))

#cora: main_acc 0.822, sub_acc 0.694
#citeseer: mian_acc 0.728  sub_acc 0.627


# rest_idx = [i for i in range(1000, features.shape[1]) if i not in idx['test']]  #433
# main_adj = np.load('Z_cora50_1_main.npy')
# W, main_embeddings = train_GCN(main_adj, idx, main_labels, main_features)
#
# rest_embeddings = get_rest_embeddings(rest_adj, rest_features, rest_labels, W)
# pred = np.argmax(main_embeddings, 1)
#
# class_embeddings = np.zeros((nclass, main_embeddings.shape[1]))
# for i in range(nclass):
#     row = np.where(pred == i)[0]
#     class_embeddings[i] = np.mean(main_embeddings[row], 0)

# Q = getQ(main_embeddings, class_embeddings, nclass)
# P = getP(Q, nclass)
# loss = cluster_loss(P, Q)
#
# criterion = F.nll_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
# lr = opt.lr
#
# print(loss)
