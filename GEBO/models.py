
"""Implements a dense version of a standard GCN model"""

import tensorflow as tf
import dgl.function as fn
ACTIVATION = tf.nn.relu


def dense_gcn_2l_tensor(placeholders, Z0, W, adj,g, hidden_dim=128, num_layer=2, dropout=None, name='dense_gcn'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()
        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        dropout = placeholders['dropout']
        W0 = W[0]
        W1 = W[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W0, dtype=tf.float32),
                                             dtype=tf.float32, trainable=True))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W1, dtype=tf.float32),
                                              dtype=tf.float32, trainable=True))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W0, dtype=tf.float32),
                                              dtype=tf.float32, trainable=True))

        # representation = ACTIVATION(hatA @ x @ W[0])
        x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        # print(x,W_par[0])
        tmp = dot(x, W_par[0], sparse=True)
        representation = ACTIVATION(A_0 @ tmp)
        # print(representation)
        # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            # representation = A_0 @ representation @ W[i]
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
        # tmp1 = dot(representation, W[-1], sparse=False)
        # out_t = dot(A_0, tmp1, sparse=False)
        # out = tf.identity(A_0 @ tmp1, 'out')
        out = tf.identity(A_0 @ representation @ W_par[-1], 'out')

        return out, W, representation

def dense_gcn_2l_tensor1(placeholders, Z0,Z1, W, adj,g, hidden_dim=16, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()
        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        A_1 = tf.cast(tf.multiply(Z1, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        # A_0 = tf.nn.softmax(A_0, dim=1)
        A_1 = constraints(A_1)
        # A_1 = tf.nn.softmax(A_1, dim=1)
        dropout = placeholders['dropout']
        W0 = W[0]
        W1 = W[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W0, dtype=tf.float32),
                                             dtype=tf.float32, trainable=True))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W1, dtype=tf.float32),
                                              dtype=tf.float32, trainable=True))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.constant(W0, dtype=tf.float32),
                                              dtype=tf.float32, trainable=True))

        # representation = ACTIVATION(hatA @ x @ W[0])
        x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        # print(x,W_par[0])
        tmp = dot(x, W_par[0], sparse=True)
        representation = ACTIVATION(A_0 @ tmp)
        # print(representation)
        # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            # representation = A_0 @ representation @ W[i]
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
        # tmp1 = dot(representation, W[-1], sparse=False)
        # out_t = dot(A_0, tmp1, sparse=False)
        # out = tf.identity(A_0 @ tmp1, 'out')
        out = tf.identity(A_1 @ representation @ W_par[-1], 'out')

        return out, W, representation


def dense_gcn_2l_tensor_test(placeholders, Z0, adj,g, hidden_dim=128,f_dim=238, out_dim=4, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        # f_dim = int(x.shape[1])
        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

        # representation = ACTIVATION(hatA @ x @ W[0])
        x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        # print(x,W_par[0])
        tmp = dot(x, W_par[0], sparse=True)
        representation = ACTIVATION(A_0 @ tmp)
        # print(representation)
        # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            # representation = A_0 @ representation @ W[i]
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
        # tmp1 = dot(representation, W[-1], sparse=False)
        # out_t = dot(A_0, tmp1, sparse=False)
        # out = tf.identity(A_0 @ tmp1, 'out')
        out = tf.identity(A_0 @ representation @ W_par[-1], 'out')

        return out, W_par, representation


def dense_gcn_2l_tensor_test1(placeholders, Z0, Z1, adj,g, hidden_dim=128,f_dim=238,out_dim=4, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        # f_dim = int(x.shape[1])
        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        A_1 = tf.cast(tf.multiply(Z1, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        A_1 = constraints(A_1)
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

        # representation = ACTIVATION(hatA @ x @ W[0])
        # x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        # print(x,W_par[0])
        tmp = dot(x, W_par[0], sparse=True)
        representation = ACTIVATION(A_0 @ tmp)
        # print(representation)
        # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            # representation = A_0 @ representation @ W[i]
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
        # tmp1 = dot(representation, W[-1], sparse=False)
        # out_t = dot(A_0, tmp1, sparse=False)
        # out = tf.identity(A_0 @ tmp1, 'out')
        out = tf.identity(A_1 @ representation @ W_par[-1], 'out')

        return out, W_par, representation

# def dense_gcn_2l_tensor_jk(placeholders, Z0, adj,g, hidden_dim=128,f_dim=238, out_dim=4, num_layer=4, dropout=None, name='JK_NET'):
#     """ Builds up a simple GCN model with dense adjacency matrix and features."""
#     with g.as_default():
#         x = placeholders['features']
#         num_features_nonzero = placeholders['num_features_nonzero']
#         adj = adj.todense()
#
#         # f_dim = int(x.shape[1])
#         try:
#             f_dim = int(x.shape[1])
#         except:
#             f_dim = f_dim
#         out_dim = out_dim
#
#         A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
#         constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
#         A_0 = constraints(A_0)
#         dropout = placeholders['dropout']
#
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             W_par = []
#             for i in range(num_layer):
#                 if i == 0:
#                     W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
#                                          shape=(f_dim, hidden_dim)))
#                 elif i == num_layer - 1:
#                     W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
#                                          shape=((num_layer-1)*hidden_dim, out_dim)))
#                 else:
#                     W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
#                                          shape=(hidden_dim, hidden_dim)))
#
#         layer = []
#         # representation = ACTIVATION(hatA @ x @ W[0])
#         # x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
#         # print(x,W_par[0])
#         tmp = dot(x, W_par[0], sparse=True)
#         representation = ACTIVATION(A_0 @ tmp)
#         layer.append(representation)
#         # print(representation)
#         # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
#         if dropout is not None:
#             representation = tf.nn.dropout(representation, 1-dropout)
#         for i in range(1, num_layer - 1):
#             representation = ACTIVATION(A_0 @ representation @ W_par[i])
#             # representation = A_0 @ representation @ W[i]
#             if dropout is not None:
#                 representation = tf.nn.dropout(representation, 1-dropout)
#             layer.append(representation)
#         # tmp1 = dot(representation, W[-1], sparse=False)
#         # out_t = dot(A_0, tmp1, sparse=False)
#         # out = tf.identity(A_0 @ tmp1, 'out')
#         out_rep = tf.concat(layer, axis=1)
#         out = tf.identity(dot(out_rep, W_par[-1]), 'out')
#
#         return out, W_par, representation


from lds_gnn.layers import *
from lds_gnn.metrics import *
import math
flags = tf.app.flags
FLAGS = flags.FLAGS

def gcn_train(placeholders, hidden_dim=128, f_dim=1433, num_layer=2, name='GCN1'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    x = placeholders['features']
    y = placeholders['labels']
    # adj = placeholders['adj']
    hatA = placeholders['support']
    try:
        f_dim = int(x.shape[1])
    except:
        f_dim = f_dim
    out_dim = int(y.shape[1])
    num_features_nonzero = placeholders['num_features_nonzero']
    dropout = placeholders['dropout']

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = []
        for i in range(num_layer):
            if i == 0:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
            elif i == num_layer - 1:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, out_dim)))
            else:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

    # if dropout is not None:
    x = sparse_dropout(x, 1 - dropout, num_features_nonzero)

    tmp = dot(x, W[0], sparse=True)
    # representation = ACTIVATION(hatA @ tmp)
    representation = ACTIVATION(dot(hatA[0], tmp, sparse=True))
    # representation = ACTIVATION(hatA @ x @ W[0])
    if dropout is not None:
        representation = tf.nn.dropout(representation, 1-dropout)
    for i in range(1, num_layer - 1):
        representation = ACTIVATION(hatA @ representation @ W[i])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
    # tmp_1 = dot(representation)
    out = tf.identity(dot(hatA[0], representation @ W[-1], sparse=True), 'out')

    return out, W, representation


def jk_train(placeholders, hidden_dim=128, f_dim=238, num_layer=5, name='JK'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    x = placeholders['features']
    y = placeholders['labels']
    # adj = placeholders['adj']
    hatA = placeholders['support']
    try:
        f_dim = int(x.shape[1])
    except:
        f_dim = f_dim
    out_dim = int(y.shape[1])
    num_features_nonzero = placeholders['num_features_nonzero']
    dropout = placeholders['dropout']

    # tilde_A = adj
    # tilde_D = tf.reduce_sum(tilde_A, axis=1)
    # sqrt_tildD = 1. / tf.sqrt(tilde_D)
    # daBda = lambda _b, _a: tf.transpose(_b * tf.transpose(_a)) * _b
    # hatA = daBda(sqrt_tildD, tilde_A)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = []
        for i in range(num_layer):
            if i == 0:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
            elif i == num_layer - 1:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=((num_layer-1)*hidden_dim, out_dim)))
            else:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

    # if dropout is not None:
    layer = []
    x = sparse_dropout(x, 1 - dropout, num_features_nonzero)

    tmp = dot(x, W[0], sparse=True)
    # representation = ACTIVATION(hatA @ tmp)
    representation = ACTIVATION(dot(hatA[0], tmp, sparse=True))
    layer.append(representation)
    # representation = ACTIVATION(hatA @ x @ W[0])
    if dropout is not None:
        representation = tf.nn.dropout(representation, 1-dropout)
    for i in range(1, num_layer - 1):
        representation = ACTIVATION(hatA @ representation @ W[i])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        layer.append(representation)
    # tmp_1 = dot(representation)
    rep = tf.concat(layer, axis=1)
    out = tf.identity(dot(rep, W[-1], sparse=False), 'out')

    return out, W, representation

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

# class GCNLayer(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  activation,
#                  dropout,
#                  bias=True):
#         super(GCNLayer, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_feats))
#         else:
#             self.bias = None
#         self.activation = activation
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = 0.
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, g, h):
#         if self.dropout:
#             h = self.dropout(h)
#         h = torch.mm(h, self.weight)
#         # normalization by square root of src degree
#         h = h * g.ndata['norm']
#         g.ndata['h'] = h
#         g.update_all(fn.copy_src(src='h', out='m'),
#                      fn.sum(msg='m', out='h'))
#         h = g.ndata.pop('h')
#         # normalization by square root of dst degree
#         h = h * g.ndata['norm']
#         # bias
#         if self.bias is not None:
#             h = h + self.bias
#         if self.activation:
#             h = self.activation(h)
#         return h
#
# class GCN_model(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(GCN_model, self).__init__()
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
#         # hidden layers
#         for i in range(n_layers - 1):
#             self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
#         # output layer
#         self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))
#
#     def forward(self, g, features):
#         h = features
#         for layer in self.layers:
#             h = layer(g, h)
#         return h
def gcn_layer(x, W, G, sparse):
    tmp = dot(x, W, sparse=sparse)

    degs = G.in_degrees().float()
    norm = tf.pow(degs, -0.5)
    norm = tf.where(tf.math.is_inf(norm), tf.zeros(norm.shape), norm)
    # norm[tf.math.is_inf(norm)] = 0
    # norm = norm.to(self.device)
    G.ndata['norm'] = tf.expand_dims(norm, -1)

    h = tmp * G.ndata['norm']
    G.ndata['h'] = h
    G.update_all(fn.copy_src(src='h', out='m'),
                 fn.sum(msg='m', out='h'))
    h = G.ndata.pop('h')
    # normalization by square root of dst degree
    h = h * G.ndata['norm']
    return h


def dense_gcn_2l_tensor_gcn(placeholders, Z0, adj,g, hidden_dim=128,f_dim=238,out_dim=4, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        # A_1 = tf.cast(tf.multiply(Z1, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        # A_1 = constraints(A_1)
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

        tmp = dot(x, W_par[0], sparse=True)

        representation = ACTIVATION(A_0 @ tmp)
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
        out = tf.identity(A_0 @ representation @ W_par[-1], 'out')

        return out, W_par


def dense_gcn_2l_tensor_gcn_m(placeholders, Z0, adj,g, hidden_dim=128,f_dim=238,out_dim=4, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        # A_0 = Z0
        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        # A_1 = tf.cast(tf.multiply(Z1, adj), tf.float32)
        # constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        # A_0 = constraints(A_0)
        # A_1 = constraints(A_1)
        # A_0 = Z0
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                             shape=(f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                             shape=(hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                             shape=(hidden_dim, hidden_dim)))


        # tmp = dot(x, W_par[0], sparse=True)
        #
        # representation = ACTIVATION(A_0 @ tmp)
        representation = x
        if dropout is not None:
            # representation = tf.nn.dropout(representation, 1-dropout)
            representation = sparse_dropout(representation, 1 - dropout, num_features_nonzero)
        representation = dot(representation, W_par[0], True)
        representation = ACTIVATION(A_0 @ representation)
        # for i in range(1, num_layer - 1):
        #     representation = ACTIVATION(A_0 @ representation @ W_par[i])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        representation = dot(representation, W_par[-1], False)
        representation = A_0 @ representation
        # out = tf.identity(A_0 @ representation @ W_par[-1], 'out')
        out = tf.identity(representation)


        return out, W_par

def dense_gcn_2l_tensor_gat(placeholders, Z, adj,g, hidden_dim=8,f_dim=238,out_dim=4, num_layer=2,K=3, dropout=None, name='GAT'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        # f_dim = int(x.shape[1])
        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        # A = []
        # constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        # for i in Z:
        #     A.append(constraints(tf.cast(tf.multiply(i, adj), tf.float32)))
        dropout = 0.4
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = []
            for i in range(len(Z)-1):
                W.append(tf.get_variable('W_%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
            W.append(tf.get_variable('W_%s' % 8, initializer=tf.glorot_uniform_initializer(),
                                     shape=((len(Z)-1)*hidden_dim, out_dim)))

        rep = []
        x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        if len(Z) == 1:
            tmp = dot(x, W[0], sparse=True)
            representation = tf.nn.elu(Z[0] @ tmp)
            # representation = tf.nn.dropout(representation, 1-dropout)
            rep.append(representation)
        else:
            for i in range(len(Z)-1):
                tmp = dot(x, W[i], sparse=True)
                representation = tf.nn.elu(Z[i] @ tmp)
                # representation = tf.nn.dropout(representation, 1-dropout)
                rep.append(representation)
            # repres = (rep[0] + rep[1] + rep[2]+ rep[3]+ rep[4]+ rep[5]+ rep[6]+ rep[7]) / 8
        if len(rep) > 1:
            repres = tf.concat(rep, axis=1)
        else:
            repres = rep[0]
        # repres = tf.nn.elu(repres)
        if dropout is not None:
            representation = tf.nn.dropout(repres, 1-dropout)

        out = tf.identity(Z[-1] @ representation @ W[-1], 'out')

        return out, W

def dense_gcn_2l_tensor_gsage(placeholders, Z0, adj,g, hidden_dim=128,f_dim=238,out_dim=4, num_layer=2, dropout=None, name='GSAGE'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()
        #adj = adj - tf.eye(adj.shape[0])

        # f_dim = int(x.shape[1])
        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        # A_1 = tf.cast(tf.multiply(Z1, adj), tf.float32)
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        A_0 = constraints(A_0)
        # A_1 = constraints(A_1)
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(num_layer*f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(num_layer*hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(num_layer*hidden_dim, hidden_dim)))
            # W_par.append(tf.get_variable('W%s' % 3, initializer=tf.glorot_uniform_initializer(),
            #                              shape=(2*hidden_dim+f_dim, out_dim)))
        # representation = ACTIVATION(hatA @ x @ W[0])
        x = sparse_dropout(x, 1 - dropout, num_features_nonzero)
        # print(x,W_par[0])
        # x = tf.sparse.reorder(x)
        x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        # x = tf.nn.dropout(x, 1-dropout)
        tmp = A_0 @ x
        # x = tf.nn.dropout(x, 1 - dropout)
        # tmp = tf.nn.dropout(tmp, 1 - dropout)
        representation = ACTIVATION(tf.concat([x, tmp], axis=1) @ W_par[0])
        # representation = representation / tf.expand_dims(tf.norm(representation, 2, 1),1)
        # tmp = dot(x, W_par[0], sparse=True)
        # representation = ACTIVATION(A_0 @ tmp)
        # print(representation.shape,x.shape)
        # rep = tf.nn.relu(tf.concat([tf.sparse.to_dense(x), representation], axis=1))
        # print(representation)
        # representation = ACTIVATION(A_0 @ placeholders['features'] @ W[0])
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        # for i in range(1, num_layer - 1):
        #     representation = ACTIVATION(A_0 @ representation @ W_par[i])
        #     # representation = A_0 @ representation @ W[i]
        #     if dropout is not None:
        #         representation = tf.nn.dropout(representation, 1-dropout)
        tmp1 = dot(A_0, representation, sparse=False)
        # tmp1 = tf.nn.dropout(tmp1, 1-dropout)
        out = ACTIVATION(tf.concat([representation, tmp1], axis=1) @ W_par[-1])
        # out = out / tf.expand_dims(tf.norm(out, 2, 1),1)
        # out_t = dot(A_1, tmp1, sparse=False)
        # out_t1 = tf.nn.relu(tf.concat([rep, out_t],1))
        # out = out_t1 @ W_par[-1]
        # out = tf.identity(A_0 @ tmp1, 'out')
        out = tf.identity(out, 'out')

        return out, W_par

def dense_gcn_2l_tensor_jk(placeholders, Z_0, adj,g, hidden_dim=128,f_dim=238, out_dim=4, num_layer=4, dropout=None, name='JK_NET'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    with g.as_default():
        x = placeholders['features']
        num_features_nonzero = placeholders['num_features_nonzero']
        adj = adj.todense()

        # f_dim = int(x.shape[1])
        try:
            f_dim = int(x.shape[1])
        except:
            f_dim = f_dim
        out_dim = out_dim

        # A_0 = tf.cast(tf.multiply(Z0, adj), tf.float32)
        # constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
        # A_0 = constraints(A_0)
        A_0 = Z_0
        dropout = placeholders['dropout']

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W_par = []
            for i in range(num_layer):
                if i == 0:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
                elif i == num_layer - 1:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=((num_layer-1)*hidden_dim, out_dim)))
                else:
                    W_par.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

        layer = []
        tmp = dot(x, W_par[0], sparse=True)
        representation = ACTIVATION(A_0 @ tmp)
        layer.append(representation)
        if dropout is not None:
            representation = tf.nn.dropout(representation, 1-dropout)
        for i in range(1, num_layer - 1):
            representation = ACTIVATION(A_0 @ representation @ W_par[i])
            if dropout is not None:
                representation = tf.nn.dropout(representation, 1-dropout)
            layer.append(representation)
        out_rep = tf.concat(layer, axis=1)
        out = tf.identity(dot(out_rep, W_par[-1]), 'out')

        return out, W_par