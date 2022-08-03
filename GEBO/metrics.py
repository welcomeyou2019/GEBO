import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
# tf.enable_eager_execution()
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def BCEWithLogitsLoss(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # print(preds, labels)
    preds = tf.gather(preds, mask)
    labels = tf.gather(labels, mask)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_mean(loss)
    # print('loss',loss)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # print('mask',mask)
    # loss *= mask
    return loss


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # pred = tf.gather(preds, mask, axis=0)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def eval_node_cls(logits, labels, mask):

    logits = tf.gather(logits, mask)
    labels = tf.gather(labels, mask)
    # print(logits, labels)
    # print('labels',logits, labels)
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
    fn = tf.reduce_sum(tf.cast(y_hat * (1 - y_true), 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_hat) * y_true, 'float'), axis=0)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # print('f1',f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)