"""THe module contains the main algorithm and a demo script to run it"""
import os
import sys
current_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.insert(0, current_path)
from lds_gnn.gcn.metrics import eval_node_cls, BCEWithLogitsLoss,masked_accuracy, masked_softmax_cross_entropy
import scipy.sparse as sp
from dgl import DGLGraph

import far_ho as far
try:
    from lds_gnn.data import ConfigData, UCI, EdgeDelConfigData
    from lds_gnn.models import dense_gcn_2l_tensor_gcn,GCN, gcn_train
    from lds_gnn.utils import *
    from lds_gnn.hyperparams import *
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from utils import *
    # noinspection PyUnresolvedReferences
    from hyperparams import *
    # noinspection PyUnresolvedReferences
    from data import ConfigData, UCI, EdgeDelConfigData
    # noinspection PyUnresolvedReferences
    from models import dense_gcn_2l_tensor_gat, gcn_train


def from_svd(svd, train=False, ss=None):
    svd['config'].train = train
    _vrs = eval('Methods.' + svd['method'])(svd['data_config'], svd['config'])
    if not train: restore_from_svd(svd, ss)
    return _vrs

class ConfigMethod(Config):
    def __init__(self, method_name=None, **kwargs):
        self.method_name = method_name
        self.seed = 1979
        self.train = True
        super().__init__(**kwargs)

    def execute(self, data_conf, **kwargs):
        return eval('Methods.' + self.method_name)(data_config=data_conf, config=self, **kwargs)


class LDSConfig(ConfigMethod):

    def __init__(self, method_name='lds', **kwargs):
        self.est_p_edge = 0.0  # initial estimation for the probability of an edge
        self.io_lr = 0.01  # learning rate for the inner optimizer (either hyperparameter sigmoid
        # or fixed value if float
        self.io_opt = 'far.AdamOptimizer'  # name of the inner objective optimizer (should be a far.Optimizer)
        self.oo_lr = 1.  # learning rate of the outer optimizer # decreasing learning rate if this is a tuple
        # self.oo_opt = 'far.GradientDescentOptimizer'  # name of the outer  objective optimizer
        self.oo_opt = 'far.AdamOptimizer'
        self.io_steps = 5  # number of steps of the inner optimization dynamics before an update (hyper batch size)
        self.io_params = (0.001, 20, 400)  # minimum decrease coeff, patience, maxiters
        self.pat = 20  # patience for early stopping
        # self.n_sample = 16  # number of samples to compute early stopping validation accuracy and test accuracy
        self.l2_reg = 5.e-4  # l2 regularization coefficient (as in Kipf, 2017 paper)
        self.keep_prob = 1.  # also this is probably not really needed
        self.num_layer = 2

        super().__init__(method_name=method_name, _version=2, **kwargs)

    def io_optimizer(self) -> far.Optimizer:
        # print(self.io_lr)
        if isinstance(self.io_lr, tuple):
            c, a, b = self.io_lr  # starting value, minimum, maximum  -> re-parametrized as a + t * (b-a) with
            # t = sigmoid(\lambda)
            lr = a + tf.sigmoid(far.get_hyperparameter('lr', -tf.log((b - a) / (c - a) - 0.99))) * (b - a)
        else:
            lr = tf.identity(self.io_lr, name='lr')
        lr = tf.identity(lr, name='io_lr')
        return eval(self.io_opt)(lr)

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
    return sparse_to_tuple(adj_normalized)

def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs

def cal_inter_par(adj, labels):
    inter = 0
    intra = 0
    row, col = np.where(adj != 0.)
    for i, j in zip(row, col):
        if labels[i] != labels[j]:
            inter += 1
        else:
            intra += 1
    return inter, intra, inter/len(row), intra/len(row), len(row)


def lds(data_conf: ConfigData, config: LDSConfig, seed):
    adj, y_train, y_val, y_val_sep, y_es, y_test, features, train_mask, val_mask, es_mask, test_mask, nclass, idx_train, idx_val, idx_test = data_conf.load()
    # print(adj)
    # adj = np.load('./adj_500.npy') + np.eye(adj.shape[0])
    # print(adj)
    setting = {'cora': [1433, 7], 'citeseer': [3703, 6], 'pubmed': [500, 3], 'airport': [238, 4], 'ppi': [50, 121],'terrorist':[106,6]}
    f_dim = setting[data_conf.dataset_name][0]
    out_dim = setting[data_conf.dataset_name][1]

    g = tf.Graph()
    with g.as_default():

        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'nclass': nclass,
            'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
            'dataset': args.d
        }

        support = [preprocess_adj(adj)]
        # weight = np.load('Z_pred_cora_50_50.npy')
        # print(weight)
        # print(cal_inter_par(weight, labels))
        # weight1 = sparse_to_tuple(sp.coo_matrix(weight))
        # support = [weight1]
        if data_conf.dataset_name == 'ppi':
            # tr_fd = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            tr_fd = construct_feed_dict(features, support, y_train, idx_train, placeholders)
            tr_fd.update({placeholders['dropout']: 0.5})

            val_fd = construct_feed_dict(features, support, y_val_sep, val_mask, placeholders)

            val_all_mask = val_mask + es_mask
            # val_all_fd = construct_feed_dict(features, support, y_val, val_all_mask, placeholders)
            val_all_fd = construct_feed_dict(features, support, y_val, idx_val, placeholders)

            # test_fd = construct_feed_dict(features, support, y_test, test_mask, placeholders)
            test_fd = construct_feed_dict(features, support, y_test, idx_test, placeholders)
        else:
            # tr_fd = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            tr_fd = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            tr_fd.update({placeholders['dropout']: 0.5})

            val_fd = construct_feed_dict(features, support, y_val_sep, val_mask, placeholders)

            val_all_mask = val_mask + es_mask
            # val_all_fd = construct_feed_dict(features, support, y_val, val_all_mask, placeholders)
            val_all_fd = construct_feed_dict(features, support, y_val, val_mask, placeholders)

            # test_fd = construct_feed_dict(features, support, y_test, test_mask, placeholders)
            test_fd = construct_feed_dict(features, support, y_test, test_mask, placeholders)

    with tf.Session(graph=g) as sess:
        tf.set_random_seed(seed)
        adj_ini = tf.convert_to_tensor(normalize_adj(adj).todense(), tf.float32)
        # adj_ini = tf.convert_to_tensor(weight, tf.float32)
        Z0 = get_stc_hyperparameter('Z0', initializer=adj_ini)#, constraints=constraint)
        # Z = [Z0] * 2
        # Z1 = get_stc_hyperparameter('Z1', initializer=adj_ini)
        # Z0 = np.array(normalize_adj(adj).todense(), dtype=np.float32)
        out, ws = dense_gcn_2l_tensor_gcn(placeholders, Z0, adj, g, num_layer=2, f_dim=f_dim, out_dim=out_dim)
        if data_conf.dataset_name == 'ppi':
            error = tf.identity(
                BCEWithLogitsLoss(out, placeholders['labels'], placeholders['labels_mask']), 'error')
            acc = tf.identity(eval_node_cls(out, placeholders['labels'], placeholders['labels_mask']), 'accuracy')
        else:
            error = tf.identity(
                masked_softmax_cross_entropy(out, placeholders['labels'], placeholders['labels_mask']), 'error')
            acc = tf.identity(masked_accuracy(out, placeholders['labels'], placeholders['labels_mask']), 'accuracy')
        tr_error = error + 5e-4 * tf.nn.l2_loss(ws[0])
        # print(error, tr_error)
        # tr_error = error

        # acc = tf.identity(eval_node_cls(out, placeholders['labels'], placeholders['labels_mask']), 'accuracy')
        # acc, tp = eval_node_cls(out, placeholders['labels'], placeholders['labels_mask'])

        svd = init_svd(data_conf, config)  # initialize data structure for saving statistics and perform early stopping
        #
        def _test_on_accept(_t):  # execute this when a better parameter configuration is found
            smp_0 = [sample(h) for h in far.utils.as_list(Z0)]
            # smp_1 = [sample(h) for h in far.utils.as_list(Z1)]
            sess.run(smp_0)
            # print(sess.run(smp_0))
            # ss.run(smp_1)
            result = sess.run(out, test_fd)
            # print(sess.run(rep, test_fd))
            # print(sess.run(out, test_fd))
            accft = sess.run(acc, {**test_fd, out: result})
            # accft = round(accft, 3)
            update_append(svd, oa_t=_t, oa_act=accft)
            # print(accft.shape)
            print('iteration', _t, ' - test accuracy: ', accft)
            global test_acc
            test_acc = accft

        def oo_optimizer(oo_lr, g, multiplier=1.):
            opt_f = far.GradientDescentOptimizer
            if isinstance(oo_lr, float):
                lr = tf.identity(multiplier * oo_lr, name='o_lrd')
                return opt_f(lr)
            elif isinstance(oo_lr, tuple):
                gs = new_gs(g)
                lrd = tf.train.inverse_time_decay(multiplier * oo_lr[0], gs, oo_lr[1],
                                                  oo_lr[2], name='o_lrd')
                return opt_f(lrd)
            else:
                raise AttributeError('not understood')

        # initialize hyperparamter optimization method
        ho_mod = far.HyperOptimizer(StcReverseHG())
        # io_opt, oo_opt = config.io_optimizer(), config.oo_optimizer()
        io_opt = config.io_optimizer()
        oo_opt = oo_optimizer(config.oo_lr, g)
        ho_step = ho_mod.minimize(error, oo_opt, tr_error, io_opt, global_step=get_gs())
        # print(ho_step)

        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # run the method
        sess.run(tf.global_variables_initializer())
        es_gen = early_stopping_with_save(config.pat, sess, svd, on_accept=_test_on_accept)
        count = 0
        best_acc = {'val_acc': 0, 'test_acc':0}
        train_loss=[]
        test_acc=[]
        if config.train:
            try:
                for _ in es_gen:  # outer optimization loop
                    print(count)
                    count+=1
                    smp_0 = [sample(h) for h in far.utils.as_list(Z0)]
                    # smp_1 = [sample(h) for h in far.utils.as_list(Z1)]
                    a = sess.run(smp_0)
                    # print(a[0])
                    # print(np.sum(a[0],1))
                    # ss.run(smp_1)
                    # result = sess.run(out, es_fd)
                    # a_es = sess.run(acc, {**es_fd, out: result})
                    # e_es = sess.run(error, {**es_fd, out: result})
                    # a_es = round(a_es,3)

                    # result = sess.run(out, val_all_fd)
                    # a_es = sess.run(acc, {**val_all_fd, out: result})
                    # # tp = sess.run(tp, {**val_all_fd, out: result})
                    # e_es = sess.run(error, {**val_all_fd, out: result})

                    result = sess.run(out, tr_fd)
                    a_es = sess.run(acc, {**tr_fd, out: result})
                    # tp = sess.run(tp, {**val_all_fd, out: result})
                    e_es = sess.run(error, {**tr_fd, out: result})
                    print(a_es)


                    # a_es = round(a_es, 3)
                    # print('tp:', tp)
                    # print('先到这里')
                    es_gen.send(a_es)  # new early stopping accuracy



                    # records some statistics -------
                    etr, atr = sess.run([error, acc], tr_fd)
                    # eva, ava = sess.run([error, acc], val_fd)
                    # eva, ava = sess.run([error, acc], val_all_fd)
                    ete, ate = sess.run([error, acc], test_fd)

                    train_loss.append(etr)
                    test_acc.append(ate)
                    print(train_loss, test_acc)

                    iolr = sess.run(io_opt.optimizer_params_tensor)
                    # print(iolr)
                    # n_edgs = np.sum(adj.eval())

                    # update_append(svd, etr=etr, atr=atr, eva=eva, ava=ava, ete=ete, ate=ate, iolr=iolr,
                    #               e_es=e_es, e_ac=a_es, olr=sess.run('o_lrd:0'))
                    # end record ----------------------

                    steps, rs = ho_mod.hypergradient.min_decrease_condition(
                        config.io_params[0], config.io_params[1], config.io_steps,
                        feed_dicts=tr_fd, verbose=False, obj=None)

                    for j in range(config.io_params[2] // config.io_steps):  # inner optimization loop
                        if rs['pat'] == 0:
                            break  # inner objective is no longer decreasing
                        ho_step(steps(), tr_fd, tr_fd, online=j)  # do one hypergradient optimization step

                        # result = sess.run(out, val_all_fd)
                        # val_acc1 = sess.run(acc, {**val_all_fd, out: result})
                        # test_acc1 = sess.run(acc, {**test_fd, out: result})
                        # if val_acc1 > best_acc['val_acc']:
                        #     best_acc['val_acc'] = val_acc1
                        #     best_acc['test_acc'] = test_acc1
                        # # print(a_es)
                        # # es_gen.send(a_es)
                        #     print(best_acc)

            except KeyboardInterrupt:
                print('Interrupted.', file=sys.stderr)
                return vars()
        return svd['es final value'], test_acc, best_acc, a

def train_GCN(dataset_name, data_conf, seed):
    ss = setup_tf(seed)

    adj, y_train, y_val, y_val_sep, y_es, y_test, features, train_mask, val_mask, es_mask, test_mask, nclass, idx_train, idx_val, idx_test = data_conf.load()

    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'nclass': nclass,
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'dataset': dataset_name
    }

    support = [preprocess_adj(adj)]

    tr_fd = construct_feed_dict(features, support, y_train, idx_train, placeholders)
    tr_fd.update({placeholders['dropout']: 0.5})

    test_fd = construct_feed_dict(features, support, y_test, idx_test, placeholders)

    val_all_mask = val_mask + es_mask
    val_all_fd = construct_feed_dict(features, support, y_val, idx_val, placeholders)
    # W = train_GCN(placeholders, tr_fd, val_all_fd, test_fd, ss)
    feature_dim = features[2][1]
    out, ws, rep = gcn_train(placeholders, f_dim=feature_dim, num_layer=2)
    error = tf.identity(
        BCEWithLogitsLoss(out, placeholders['labels'], placeholders['labels_mask']), 'error')
    tr_error = error + 5e-4 * tf.nn.l2_loss(ws[0])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    acc = tf.identity(eval_node_cls(out, placeholders['labels'], placeholders['labels_mask']), 'accuracy')

    opt_op = optimizer.minimize(tr_error)

    best_val_acc = 0
    best_test_acc = 0
    best_test_loss = 0

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

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            W = out_tr[-1]
            best_test_acc = acc_test
            best_test_loss = loss_test

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_tr),
              "train_acc=", "{:.5f}".format(acc_tr), "val_loss=", "{:.5f}".format(loss_val),
              "val_acc=", "{:.5f}".format(acc_val))

    print("Final results:", "cost=", "{:.5f}".format(best_test_loss),
          "accuracy=", "{:.5f}".format(best_test_acc))
    return best_test_acc

def main(data, method, seed, missing_percentage):
    data_config = EdgeDelConfigData(prob_del=missing_percentage, seed=seed, enforce_connected=False,
                                        dataset_name=data)

    if method == 'lds':
        configs = LDSConfig.grid(pat=20, seed=seed, io_steps=[1],
                                 io_lr=0.01, keep_prob=0.5,
                                 oo_lr=0.99)
                                 # oo_lr=[(.1, 1., 1.e-1)])
                                 # oo_lr=10)
    else:
        raise NotImplementedError('Method {} unknown. Choose between `knnlds` and `lds`'.format(method))


    # gcn_acc = []
    # for i in range(5):
    #     seed = np.random.randint(0, 1e10)
    #
    #     test_acc = train_GCN(args.d, data_config, seed)
    #     gcn_acc.append(test_acc)
    # print(np.mean(gcn_acc), np.std(gcn_acc))

    # W=0
    result = {5: [], 10: [], 15: [], 20: [], 25: []}
    for i in range(5):

        seed = np.random.randint(0, 1e10)
        print('Round %d'%(i))
        for cnf in configs:
            best_valid_acc = 0
            best_test_acc = 0
            print('cnf', cnf)
            valid_acc, test_acc, best_acc, modify_adj = lds(data_config, cnf, seed)
            if best_valid_acc <= valid_acc:
                # print('Found a better configuration:', valid_acc)
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                # np.save('Z_jk' + data_config.dataset_name + str(cnf.io_steps)+'_'+str(i) +'.npy', modify_adj[0])

            print('Test accuracy of the best found model:', best_test_acc)
            result[cnf.io_steps].append(best_acc['test_acc'])
            print(result)

    result_mean = []
    result_std = []
    for i in result:
        result_mean.append(np.mean(result[i]))
        result_std.append(np.std(result[i]))
    print('mean：', result_mean)
    print('std:', result_std)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('-d', default='cora', type=str,
                        help='The evaluation dataset:cora, citeseer, airport. blogcatalog: flickr，terrorist')
    parser.add_argument('-m', default='lds', type=str,
                        help='The method: lds or knnlds. Default: knnlds')
    parser.add_argument('-s', default=1, type=int,
                        help='The random seed. Default: 1')
    parser.add_argument('-e', default=0, type=int,
                        help='The percentage of missing edges (valid only for cora and citeseer dataset): Default 50. - ' +
                             'PLEASE NOTE THAT the x-axes of Fig. 2  in the paper reports the percentage of retained edges rather ' +
                             'than that of missing edges.')
    args = parser.parse_args()
    for i in ['cora']:
        _data, _method, _seed, _missing_percentage = i, args.m, args.s, args.e / 100

        main(_data, _method, _seed, _missing_percentage)
