
"""The module contains the main class responsible for hypergradient computation (estimation)
as well as utility functions to create and manage hyperparamter variables"""

import tensorflow as tf
import far_ho as far

try:
    import lds_gnn.utils
    from lds_gnn.utils import GraphKeys
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from utils import GraphKeys

_STC_INITIALIZERs = {}
_STC_MAP = {}


def symm_adj_sample(probs):
    """Sampling function for symmetric Bernoulli matrices"""
    e = bernoulli_hard_sample(probs)
    return e + tf.transpose(e)


def bernoulli_hard_sample(probs):
    """Sampling function for Bernoulli"""
    return tf.floor(tf.random_uniform(probs.shape, minval=0., maxval=1.) + probs)


def get_stc_hyperparameter(name, initializer=None, shape=None, constraints=None,
                           sample_func=None, hyper_probs=None):
    """
    Get a stochastic hyperparameter. Defaults to Bernoulli hyperparameter. Mostly follows the signature of
    `tf.get_variable`

    :param name: a name for the hyperparameter
    :param initializer: an initializer (or initial value) for the parameters of the distribution
    :param shape: a shape for the stochastic hyperparameter
    :param constraints: additional (simple) constraints for the parameters of the distribution
    :param sample_func: a function that takes the distribution parameters and returns a sample
    :param hyper_probs: the variables used for the underlying probability distribution
    :return: The stochastic hyperparameter (not the distribution variables!)
    """
    if constraints is None:
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
    if hyper_probs is None:  # 初始化A
        hyper_probs = tf.get_variable(
            name + '/' + GraphKeys.STOCHASTIC_HYPER, trainable=False,
            # constraint=constraints,
            initializer=initializer,
            shape=shape,
            collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.STOCHASTIC_HYPER],
            dtype=tf.float32
        )
    if sample_func is None:
        sample_func = tf.random_uniform
    hyper_sample = far.get_hyperparameter(
        name,
        initializer=hyper_probs,#+tf.transpose(hyper_probs)-tf.matrix_diag(tf.diag_part(hyper_probs)),
        collections=GraphKeys.STOCHASTIC_HYPER,
        dtype=tf.float32
    )
    far.utils.remove_from_collection(GraphKeys.GLOBAL_VARIABLES, hyper_sample)
    with tf.control_dependencies([tf.variables_initializer([hyper_sample])]):  # re-initialize and return the value
        _STC_INITIALIZERs[hyper_sample] = hyper_sample.read_value()

    _STC_MAP[hyper_sample] = hyper_probs

    return hyper_sample

def get_stc_hyperparameter1(name, initializer=None, shape=None, constraints=None,
                           sample_func=None, hyper_probs=None):
    """
    Get a stochastic hyperparameter. Defaults to Bernoulli hyperparameter. Mostly follows the signature of
    `tf.get_variable`

    :param name: a name for the hyperparameter
    :param initializer: an initializer (or initial value) for the parameters of the distribution
    :param shape: a shape for the stochastic hyperparameter
    :param constraints: additional (simple) constraints for the parameters of the distribution
    :param sample_func: a function that takes the distribution parameters and returns a sample
    :param hyper_probs: the variables used for the underlying probability distribution
    :return: The stochastic hyperparameter (not the distribution variables!)
    """
    if constraints is None:
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
    if hyper_probs is None:  # 初始化A
        hyper_probs = tf.get_variable(
            name + '/' + GraphKeys.STOCHASTIC_HYPER, trainable=False,
            constraint=constraints,
            initializer=initializer,
            shape=shape,
            collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.STOCHASTIC_HYPER],
            dtype=tf.float32
        )
    if sample_func is None:
        sample_func = tf.random_uniform
    hyper_sample = far.get_hyperparameter(
        name,
        initializer=hyper_probs+tf.transpose(hyper_probs)-tf.matrix_diag(tf.diag_part(hyper_probs)),
        collections=GraphKeys.STOCHASTIC_HYPER,
        dtype=tf.float32
    )
    far.utils.remove_from_collection(GraphKeys.GLOBAL_VARIABLES, hyper_sample)
    with tf.control_dependencies([tf.variables_initializer([hyper_sample])]):  # re-initialize and return the value
        _STC_INITIALIZERs[hyper_sample] = hyper_sample.read_value()

    _STC_MAP[hyper_sample] = hyper_probs

    return hyper_sample

get_bernoulli_hyperparameter = get_stc_hyperparameter


def get_probs_var(hyper):
    """Returns the distribution's parameters of stochastic hyperparameter"""
    # print(hyper)
    # print(_STC_MAP)
    return _STC_MAP[hyper]


def sample(hyper):
    """ Returns a `sampler` operation (in the form of an initializer, for the stochastic hyperparameter `hyper`"""
    return _STC_INITIALIZERs[hyper]


def is_stochastic_hyper(hyper):
    """Returns true if the hyperparameter is stochastic"""
    return hyper in tf.get_collection(GraphKeys.STOCHASTIC_HYPER)


def hyper_or_stochastic_hyper(hyper):
    """Returns either the underlying parameters of the probability distribution if `hyper` is stochastic,
    or `hyper`"""
    return get_probs_var(hyper) if is_stochastic_hyper(hyper) else hyper


class StcReverseHG(far.ReverseHG):
    """
    Subclass of `far.ReverseHG` that deals also with stochastic hyperparameters
    """

    def __init__(self, history=None, name='ReverseHGPlus'):
        super().__init__(history, name)
        self.samples = None

    @property
    def initialization(self):
        if self._initialization is None:
            additional_initialization = [sample(h) for h in self._hypergrad_dictionary
                                         if is_stochastic_hyper(h)]
            self.samples = additional_initialization
            # noinspection PyStatementEffect
            super(StcReverseHG, self).initialization
            self._initialization.extend(additional_initialization)
        return super(StcReverseHG, self).initialization

    def hgrads_hvars(self, hyper_list=None, aggregation_fn=None, process_fn=None):
        rs = super(StcReverseHG, self).hgrads_hvars(hyper_list, aggregation_fn, process_fn)

        def _ok_or_store_var(hg_hv_pair):
            if is_stochastic_hyper(hg_hv_pair[1]):
                return hg_hv_pair[0], get_probs_var(hg_hv_pair[1])
            return hg_hv_pair

        return [_ok_or_store_var(pair) for pair in rs]

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):
        """As in `far.ReverseHG.run`, plus, samples the stochastic hyperparameters at every iterations
        of the inner optimization dynamics"""
        cbk_multi_sample = lambda _t, _fd, _ss: _ss.run(self.samples)
        super().run(T_or_generator, inner_objective_feed_dicts, outer_objective_feed_dicts, initializer_feed_dict,
                    global_step, session, online, [cbk_multi_sample, cbk_multi_sample])

    def hypergradient(self, hyper):
        hg = self._hypergrad_dictionary[hyper]
        return hg[0] if len(hg) == 1 else hg

    @staticmethod
    def _create_hypergradient(outer_obj, hyper):
        doo_dh = tf.gradients(outer_obj, hyper)[0]
        doo_dh = far.utils.val_or_zero(doo_dh, hyper)
        if is_stochastic_hyper(hyper):
            doo_dh = far.utils.maybe_add(doo_dh, tf.gradients(outer_obj, get_probs_var(hyper))[0])
        return far.ReverseHG._create_hypergradient_from_dodh(hyper, doo_dh)

    def min_decrease_condition(self, dec=.001, patience=20, max_iters=1e6,
                               session=None, feed_dicts=None, verbose=False, obj=None, auto=True):
        """Step generator that takes into account the "decrease condition" (of the inner objective) to
        stop inner objective optimization"""
        if obj is None:
            obj = list(self._optimizer_dicts)[0].objective  #优化tr_error
            # print(list(self._optimizer_dicts)[0])
            # print(obj)
        if session is None: session = tf.get_default_session()
        res_dic = {'pat': patience, 'min val': None}

        def _gen(p0=patience, val0=None):
            t = 0
            p = p0
            if val0 is None:
                prev_val = val = session.run(obj, feed_dicts)
            else:
                prev_val = val = val0
            while p > 0 and t < max_iters:
                val = session.run(obj, feed_dicts)
                if verbose > 1: print(t, 'min MD condition', prev_val, val, 'pat:', p)
                if prev_val * (1. - dec) < val:
                    p -= 1
                else:
                    p = patience
                    prev_val = val
                yield t
                t += 1
            res_dic.update({'pat': p, 'val': val, 'min val': prev_val, 'tot iter': t})
            if verbose: print(res_dic)

        if auto:
            def _the_gens():
                return _gen(res_dic['pat'], res_dic['min val']), range(int(max_iters) + 1)
        else:
            def _the_gens(p0=patience, val0=None):
                return _gen(p0, val0), range(
                    int(max_iters) + 1)

        return _the_gens, res_dic
