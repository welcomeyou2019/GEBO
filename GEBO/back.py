"""THe module contains the main algorithm and a demo script to run it"""
import networkx as nx
import gcn.metrics
from sklearn.neighbors import kneighbors_graph
import far_ho as far
# from gcn.utils import load_data
# from .hyperparams import StcReverseHG
# import tensorflow as tf
# import numpy as np
import scipy.sparse as sp
import numpy as np

try:
    from lds_gnn.data import ConfigData, UCI, EdgeDelConfigData
    from lds_gnn.models import dense_gcn_2l_tensor
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
    from models import dense_gcn_2l_tensor


def from_svd(svd, train=False, ss=None):
    svd['config'].train = train
    _vrs = eval('Methods.' + svd['method'])(svd['data_config'], svd['config'])
    if not train: restore_from_svd(svd, ss)
    return _vrs

# def get_param():
#     saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
#     with tf.Session() as sess1:
#         saver.restore(sess1, tf.train.latest_checkpoint('./checkpoint_dir/'))  # 加载模型变量
#         graph = tf.get_default_graph()
#         W_0 = graph.get_tensor_by_name("gcn/graphconvolution_1_vars/weights_0:0")  # 根据tensor名字获取tensor变量
#         W_1 = graph.get_tensor_by_name("gcn/graphconvolution_2_vars/weights_0:0")
#         a = W_0.eval()
#         b = W_1.eval()
#         tf.get_default_graph().close()
#     print(a,b)
#     return a,b

def empirical_mean_model(S, sample_vars, model_out, *what, fd=None, ss=None):
    """ Computes the tensors in `what` using the empirical mean output of the model given by
    `model_out`, sampling `S` times the stochastic variables in `sample_vars`"""
    if ss is None: ss = tf.get_default_session()
    smp = [sample(h) for h in far.utils.as_list(sample_vars)]
    mean_out = []
    for i in range(S):
        ss.run(smp)
        mean_out.append(ss.run(model_out, fd))
    mean_out = np.mean(mean_out, axis=0)
    lst = ss.run(what, {**fd, model_out: mean_out})
    return lst[0] if len(lst) == 1 else lst


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
        self.oo_opt = 'far.GradientDescentOptimizer'  # name of the outer  objective optimizer
        self.io_steps = 5  # number of steps of the inner optimization dynamics before an update (hyper batch size)
        self.io_params = (0.001, 20, 400)  # minimum decrease coeff, patience, maxiters
        self.pat = 20  # patience for early stopping
        # self.n_sample = 16  # number of samples to compute early stopping validation accuracy and test accuracy
        self.l2_reg = 5.e-4  # l2 regularization coefficient (as in Kipf, 2017 paper)
        self.keep_prob = 1.  # also this is probably not really needed
        self.num_layer = 2

        super().__init__(method_name=method_name, _version=2, **kwargs)

    def io_optimizer(self) -> far.Optimizer:
        if isinstance(self.io_lr, tuple):
            c, a, b = self.io_lr  # starting value, minimum, maximum  -> re-parametrized as a + t * (b-a) with
            # t = sigmoid(\lambda)
            lr = a + tf.sigmoid(far.get_hyperparameter('lr', -tf.log((b - a) / (c - a) - 0.99))) * (b - a)
        else:
            lr = tf.identity(self.io_lr, name='lr')
        lr = tf.identity(lr, name='io_lr')
        return eval(self.io_opt)(lr)

    def oo_optimizer(self, multiplier=1.):
        opt_f = eval(self.oo_opt)
        if isinstance(self.oo_lr, float):
            lr = tf.identity(multiplier * self.oo_lr, name='o_lrd')
            return opt_f(lr)
        elif isinstance(self.oo_lr, tuple):
            gs = new_gs()
            lrd = tf.train.inverse_time_decay(multiplier * self.oo_lr[0], gs, self.oo_lr[1],
                                              self.oo_lr[2], name='o_lrd')
            return opt_f(lrd)
        else:
            raise AttributeError('not understood')


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
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


class KNNLDSConfig(LDSConfig):
    def __init__(self, method_name='lds', **kwargs):
        """Configuration instance for the method kNN-LDS.
        The arguments are the same as LDSConfig plus

        `k` (10): number of neighbours

        `metric` (cosine): metric function to use"""
        self.k = 10
        self.metric = 'cosine'
        super().__init__(method_name, **kwargs)


def cal_subgraph(adj):
    G = nx.from_numpy_array(adj)
    subgraph = nx.connected_components(G)
    return subgraph


def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs


def lds(data_conf: ConfigData, config: LDSConfig):
    """
    Runs the LDS algorithm on data specified by `data_conf` with parameters
    specified in `config`.

    :param data_conf: Configuration for the data. Please see `ConfigData` for documentation
    :param config: Configuration for the method's parameters. Please see `LDSConfig` for documentation
    :return: a triplet: - the local variable dictionary (as returned by vars()),
                        - the best `early stopping` accuracy
                        - the test accuracy on the iteration that achieved the best `early stopping` accuracy
    """
    ss = setup_tf(config.seed)

    adj, adj_mods, features, ys, train_mask, val_mask, es_mask, test_mask = data_conf.load()

    plc = Placeholders(features, ys)

    constraint = upper_tri_const(adj.shape)  # 上三角变为0，1之间
    W0 = np.loadtxt('./parameter/' + data_conf.dataset_name + '_W0')
    W1 = np.loadtxt('./parameter/' + data_conf.dataset_name + '_W1')
    adj_ini = tf.cast(preprocess_adj(adj), tf.float32)

    # Z0 = get_stc_hyperparameter('Z0', initializer=constraint(adj_ini), constraints=constraint)
    Z0 = get_stc_hyperparameter('Z0', initializer=adj_ini)
    # smp_0 = [sample(h) for h in far.utils.as_list(Z0)]
    # print(Z0.eval())
    # print(len(smp_0))
    # for i in smp_0:
    #     print(i.eval())
    # Z1 = get_stc_hyperparameter('Z1', initializer=adj_ini)
    # out, ws, rep = dense_gcn_2l_tensor(plc.X, adj_hyp, plc.Y, num_layer=config.num_layer,
    #                                    dropout=plc.keep_prob)

    out, ws, rep = dense_gcn_2l_tensor(plc.X, adj, plc.Y, Z0, W0, W1, num_layer=config.num_layer,
                                       dropout=plc.keep_prob)

    error = tf.identity(gcn.metrics.masked_softmax_cross_entropy(out, plc.Y, plc.label_mask), 'error')
    tr_error = error + config.l2_reg * tf.nn.l2_loss(ws[0])

    acc = tf.identity(gcn.metrics.masked_accuracy(out, plc.Y, plc.label_mask), 'accuracy')

    tr_fd, val_fd, es_fd, test_fd = plc.fds(train_mask, val_mask, es_mask, test_mask)
    tr_fd = {**tr_fd, **{plc.keep_prob: config.keep_prob}}

    svd = init_svd(data_conf, config)  # initialize data structure for saving statistics and perform early stopping

    def _test_on_accept(_t):  # execute this when a better parameter configuration is found
        # accft = empirical_mean_model(config.n_sample, adj, out, acc, fd=test_fd)
        ss = tf.get_default_session()
        smp_0 = [sample(h) for h in far.utils.as_list(Z0)]
        # smp_1 = [sample(h) for h in far.utils.as_list(Z1)]
        ss.run(smp_0)
        # ss.run(smp_1)
        result = ss.run(out, test_fd)
        accft = ss.run(acc, {**test_fd, out: result})
        accft = round(accft, 3)
        update_append(svd, oa_t=_t, oa_act=accft)
        print('iteration', _t, ' - test accuracy: ', accft)
        global test_acc
        test_acc = accft

    # initialize hyperparamter optimization method
    ho_mod = far.HyperOptimizer(StcReverseHG())
    io_opt, oo_opt = config.io_optimizer(), config.oo_optimizer()
    ho_step = ho_mod.minimize(error, oo_opt, tr_error, io_opt, global_step=get_gs())

    # run the method
    tf.global_variables_initializer().run()
    es_gen = early_stopping_with_save(config.pat, ss, svd, on_accept=_test_on_accept)
    count = 0
    if config.train:
        try:
            for _ in es_gen:  # outer optimization loop
                print(count)
                count+=1
                # e_es, a_es = empirical_mean_model(config.n_sample, adj_hyp, out, error, acc, fd=es_fd)

                smp_0 = [sample(h) for h in far.utils.as_list(Z0)]
                # smp_1 = [sample(h) for h in far.utils.as_list(Z1)]
                ss.run(smp_0)
                # ss.run(smp_1)
                result = ss.run(out, es_fd)
                a_es = ss.run(acc, {**es_fd, out: result})
                e_es = ss.run(error, {**es_fd, out: result})
                a_es = round(a_es,3)

                es_gen.send(a_es)  # new early stopping accuracy

                # records some statistics -------
                etr, atr = ss.run([error, acc], tr_fd)
                eva, ava = ss.run([error, acc], val_fd)
                ete, ate = ss.run([error, acc], test_fd)
                iolr = ss.run(io_opt.optimizer_params_tensor)
                # n_edgs = np.sum(adj.eval())

                update_append(svd, etr=etr, atr=atr, eva=eva, ava=ava, ete=ete, ate=ate, iolr=iolr,
                              e_es=e_es, e_ac=a_es, olr=ss.run('o_lrd:0'))
                # end record ----------------------

                steps, rs = ho_mod.hypergradient.min_decrease_condition(
                    config.io_params[0], config.io_params[1], config.io_steps,
                    feed_dicts=tr_fd, verbose=False, obj=None)
                for j in range(config.io_params[2] // config.io_steps):  # inner optimization loop
                    if rs['pat'] == 0: break  # inner objective is no longer decreasing
                    ho_step(steps(), tr_fd, val_fd, online=j)  # do one hypergradient optimization step
        except KeyboardInterrupt:
            print('Interrupted.', file=sys.stderr)
            return vars()
    return vars(), svd['es final value'], test_acc


def main(data, method, seed, missing_percentage):
    if data == 'iris':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)
    elif data == 'wine':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)
    elif data == 'breast_cancer':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)
    elif data == 'digits':
        data_config = UCI(seed=seed, dataset_name=data, n_train=50, n_val=50, n_es=50, scale=False)
    elif data == '20newstrain':
        data_config = UCI(seed=seed, dataset_name=data, n_train=200, n_val=200, n_es=200, scale=False)
    elif data == '20news10':
        data_config = UCI(seed=seed, dataset_name=data, n_train=100, n_val=100, n_es=100, scale=False)
    elif data == 'cora' or data == 'citeseer' or data == 'ppi' or data == 'blogcatalog' or data == 'airport' or data == 'flickr':
        data_config = EdgeDelConfigData(prob_del=missing_percentage, seed=seed, enforce_connected=False,
                                        dataset_name=data)
    elif data == 'fma':
        data_config = UCI(seed=seed, dataset_name=data, n_train=160, n_val=160, n_es=160, scale=False)
    else:
        raise AttributeError('Dataset {} not available'.format(data))

    if method == 'knnlds':
        configs = KNNLDSConfig.grid(pat=20, seed=seed, io_steps=[5, 16], keep_prob=0.5,
                                    io_lr=(2.e-2, 1.e-4, 0.05),
                                    oo_lr=[(1., 1., 1.e-3), (.1, 1., 1.e-3)],
                                    metric=['cosine', 'minkowski'], k=[10, 20])
    elif method == 'lds':
        # configs = LDSConfig.grid(pat=20, seed=seed, io_steps=[20],
        #                          io_lr=(2.e-2, 1.e-4, 0.05), keep_prob=0.5,
        #                          oo_lr=[(.1, 1., 1.e-3)])
        configs = LDSConfig.grid(pat=20, seed=seed, io_steps=[5, 10, 15, 20, 25],
                                 io_lr=0.01, keep_prob=0.5,
                                 oo_lr=[(.1, 1., 1.e-3)])
    else:
        raise NotImplementedError('Method {} unknown. Choose between `knnlds` and `lds`'.format(method))

    best_valid_acc = 0
    best_test_acc = 0
    print('data_config', configs)
    for cnf in configs:
        print('cnf', cnf)
        vrs, valid_acc, test_acc = lds(data_config, cnf)
        if best_valid_acc <= valid_acc:
            print('Found a better configuration:', valid_acc)
            best_valid_acc = valid_acc
            best_test_acc = test_acc

        print('Test accuracy of the best found model:', best_test_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('-d', default='airport', type=str,
                        help='The evaluation dataset: iris, wine, breast_cancer, digits, 20newstrain, 20news10, ' +
                             'cora, citeseer, airport. blogcatalog: flickr, ppi')
    parser.add_argument('-m', default='lds', type=str,
                        help='The method: lds or knnlds. Default: knnlds')
    parser.add_argument('-s', default=1, type=int,
                        help='The random seed. Default: 1')
    parser.add_argument('-e', default=0, type=int,
                        help='The percentage of missing edges (valid only for cora and citeseer dataset): Default 50. - ' +
                             'PLEASE NOTE THAT the x-axes of Fig. 2  in the paper reports the percentage of retained edges rather ' +
                             'than that of missing edges.')
    args = parser.parse_args()

    _data, _method, _seed, _missing_percentage = args.d, args.m, args.s, args.e / 100

    main(_data, _method, _seed, _missing_percentage)
