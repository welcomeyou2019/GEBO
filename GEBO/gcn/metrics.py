import tensorflow as tf
# from sklearn.metrics import f1_score
# import tensorflow_addons as tfa

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


# def masked_accuracy(preds, labels, mask):
#     """Accuracy with masking."""
#     correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
#     accuracy_all = tf.cast(correct_prediction, tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     mask /= tf.reduce_mean(mask)
#     accuracy_all *= mask
#     return tf.reduce_mean(accuracy_all)

def BCEWithLogitsLoss(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # print(preds, labels)
    preds = tf.gather(preds, mask)
    labels = tf.gather(labels, mask)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_mean(loss)
    return loss


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def eval_node_cls(logits, labels, mask):

    logits = tf.gather(logits, mask)
    labels = tf.gather(labels, mask)
    preds = tf.round(tf.nn.sigmoid(logits))
    micro_f1 = f1(preds, labels)
    # print(micro_f1)
    # calc confusion matrix
    # conf_mat = np.zeros((self.n_class, self.n_class))
    # for i in range(len(preds)):
    #     conf_mat[labels[i], preds[i]] += 1
    return micro_f1


def f1(y_hat, y_true, model='multi'):
    '''
    输入张量y_hat是输出层经过sigmoid激活的张量
    y_true是label{0,1}的集和
    model指的是如果是多任务分类，single会返回每个分类的f1分数，multi会返回所有类的平均f1分数（Marco-F1）
    如果只是单个二分类任务，则可以忽略model
    '''
    epsilon = 1e-7
    # y_hat = tf.round(y_hat)  # 将经过sigmoid激活的张量四舍五入变为0，1输出

    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    # tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast(y_hat, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true, 'float'), axis=0)

    p = tf.reduce_sum(tp) / (tf.reduce_sum(fp) + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tf.reduce_sum(tp) / (tf.reduce_sum(fn) + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # print('f1',f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)
        # return f1

# def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None,
#                                     pos_label=1, average=None,
#                                     warn_for=('precision', 'recall',
#                                               'f-score'),
#                                     sample_weight=None,
#                                     zero_division="warn"):
#
#     # Calculate tp_sum, pred_sum, true_sum ###
#     samplewise = average == 'samples'
#     MCM = multilabel_confusion_matrix(y_true, y_pred,
#                                       sample_weight=sample_weight,
#                                       labels=labels, samplewise=samplewise)
#     tp_sum = MCM[:, 1, 1]
#     pred_sum = tp_sum + MCM[:, 0, 1]
#     true_sum = tp_sum + MCM[:, 1, 0]
#
#     if average == 'micro':
#         tp_sum = np.array([tp_sum.sum()])
#         pred_sum = np.array([pred_sum.sum()])
#         true_sum = np.array([true_sum.sum()])
#
#     # Finally, we have all our sufficient statistics. Divide! #
#     beta2 = beta ** 2
#
#     # Divide, and on zero-division, set scores and/or warn according to
#     # zero_division:
#     precision = _prf_divide(tp_sum, pred_sum, 'precision',
#                             'predicted', average, warn_for, zero_division)
#     recall = _prf_divide(tp_sum, true_sum, 'recall',
#                          'true', average, warn_for, zero_division)
#
#     # warn for f-score only if zero_division is warn, it is in warn_for
#     # and BOTH prec and rec are ill-defined
#     if zero_division == "warn" and ("f-score",) == warn_for:
#         if (pred_sum[true_sum == 0] == 0).any():
#             _warn_prf(
#                 average, "true nor predicted", 'F-score is', len(true_sum)
#             )
#
#     # if tp == 0 F will be 1 only if all predictions are zero, all labels are
#     # zero, and zero_division=1. In all other case, 0
#     if np.isposinf(beta):
#         f_score = recall
#     else:
#         denom = beta2 * precision + recall
#
#         denom[denom == 0.] = 1  # avoid division by 0
#         f_score = (1 + beta2) * precision * recall / denom
#
#     # Average the results
#     if average == 'weighted':
#         weights = true_sum
#         if weights.sum() == 0:
#             zero_division_value = np.float64(1.0)
#             if zero_division in ["warn", 0]:
#                 zero_division_value = np.float64(0.0)
#             # precision is zero_division if there are no positive predictions
#             # recall is zero_division if there are no positive labels
#             # fscore is zero_division if all labels AND predictions are
#             # negative
#             if pred_sum.sum() == 0:
#                 return (zero_division_value,
#                         zero_division_value,
#                         zero_division_value,
#                         None)
#             else:
#                 return (np.float64(0.0),
#                         zero_division_value,
#                         np.float64(0.0),
#                         None)
#
#     elif average == 'samples':
#         weights = sample_weight
#     else:
#         weights = None
#
#     if average is not None:
#         assert average != 'binary' or len(precision) == 1
#         precision = np.average(precision, weights=weights)
#         recall = np.average(recall, weights=weights)
#         f_score = np.average(f_score, weights=weights)
#         true_sum = None  # return no support
#
#     return precision, recall, f_score, true_sum
#
# def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None,
#                                 labels=None, samplewise=False):
#
#     present_labels = unique_labels(y_true, y_pred)
#     if labels is None:
#         labels = present_labels
#         n_labels = None
#     else:
#         n_labels = len(labels)
#         labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
#                                                  assume_unique=True)])
#
#     if y_true.ndim == 1:
#         if samplewise:
#             raise ValueError("Samplewise metrics are not available outside of "
#                              "multilabel classification.")
#
#         le = LabelEncoder()
#         le.fit(labels)
#         y_true = le.transform(y_true)
#         y_pred = le.transform(y_pred)
#         sorted_labels = le.classes_
#
#         # labels are now from 0 to len(labels) - 1 -> use bincount
#         tp = y_true == y_pred
#         tp_bins = y_true[tp]
#         if sample_weight is not None:
#             tp_bins_weights = np.asarray(sample_weight)[tp]
#         else:
#             tp_bins_weights = None
#
#         if len(tp_bins):
#             tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
#                                  minlength=len(labels))
#         else:
#             # Pathological case
#             true_sum = pred_sum = tp_sum = np.zeros(len(labels))
#         if len(y_pred):
#             pred_sum = np.bincount(y_pred, weights=sample_weight,
#                                    minlength=len(labels))
#         if len(y_true):
#             true_sum = np.bincount(y_true, weights=sample_weight,
#                                    minlength=len(labels))
#
#         # Retain only selected labels
#         indices = np.searchsorted(sorted_labels, labels[:n_labels])
#         tp_sum = tp_sum[indices]
#         true_sum = true_sum[indices]
#         pred_sum = pred_sum[indices]
#
#     else:
#         sum_axis = 1 if samplewise else 0
#
#         # All labels are index integers for multilabel.
#         # Select labels:
#         if not np.array_equal(labels, present_labels):
#             if np.max(labels) > np.max(present_labels):
#                 raise ValueError('All labels must be in [0, n labels) for '
#                                  'multilabel targets. '
#                                  'Got %d > %d' %
#                                  (np.max(labels), np.max(present_labels)))
#             if np.min(labels) < 0:
#                 raise ValueError('All labels must be in [0, n labels) for '
#                                  'multilabel targets. '
#                                  'Got %d < 0' % np.min(labels))
#
#         if n_labels is not None:
#             y_true = y_true[:, labels[:n_labels]]
#             y_pred = y_pred[:, labels[:n_labels]]
#
#         # calculate weighted counts
#         true_and_pred = y_true.multiply(y_pred)
#         tp_sum = count_nonzero(true_and_pred, axis=sum_axis,
#                                sample_weight=sample_weight)
#         pred_sum = count_nonzero(y_pred, axis=sum_axis,
#                                  sample_weight=sample_weight)
#         true_sum = count_nonzero(y_true, axis=sum_axis,
#                                  sample_weight=sample_weight)
#
#     fp = pred_sum - tp_sum
#     fn = true_sum - tp_sum
#     tp = tp_sum
#
#     if sample_weight is not None and samplewise:
#         sample_weight = np.array(sample_weight)
#         tp = np.array(tp)
#         fp = np.array(fp)
#         fn = np.array(fn)
#         tn = sample_weight * y_true.shape[1] - tp - fp - fn
#     elif sample_weight is not None:
#         tn = sum(sample_weight) - tp - fp - fn
#     elif samplewise:
#         tn = y_true.shape[1] - tp - fp - fn
#     else:
#         tn = y_true.shape[0] - tp - fp - fn
#
#     return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
#
# def unique_labels(*ys):
#     if not ys:
#         raise ValueError('No argument has been passed.')
#     # Check that we don't mix label format
#
#     ys_types = set(type_of_target(x) for x in ys)
#     if ys_types == {"binary", "multiclass"}:
#         ys_types = {"multiclass"}
#
#     if len(ys_types) > 1:
#         raise ValueError("Mix type of y not allowed, got types %s" % ys_types)
#
#     label_type = ys_types.pop()
#
#     # Check consistency for the indicator format
#     if (label_type == "multilabel-indicator" and
#             len(set(check_array(y,
#                                 accept_sparse=['csr', 'csc', 'coo']).shape[1]
#                     for y in ys)) > 1):
#         raise ValueError("Multi-label binary indicator input with "
#                          "different numbers of labels")
#
#     # Get the unique set of labels
#     _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
#     if not _unique_labels:
#         raise ValueError("Unknown label type: %s" % repr(ys))
#
#     ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))
#
#     # Check that we don't mix string type with number type
#     if (len(set(isinstance(label, str) for label in ys_labels)) > 1):
#         raise ValueError("Mix of label input types (string and number)")
#
#     return np.array(sorted(ys_labels))