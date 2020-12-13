from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def hook_cls(pred, label, name, step=-1, threshold=.5):
    b_pred = tf.cast(tf.greater(pred, threshold), tf.float32)

    g_pos = tf.cast(tf.equal(label, 1), tf.float32)
    g_neg = tf.cast(tf.equal(label, 0), tf.float32)

    p_pos = b_pred
    p_neg = 1 - b_pred

    assertion = tf.cast(tf.equal(b_pred, label), tf.float32)

    # 0. Overall
    acc = tf.reduce_mean(assertion)

    # 1. TP, recall
    tp = tf.reduce_sum(g_pos * p_pos) / tf.reduce_sum(g_pos)

    # 2. TN
    tn = tf.reduce_sum(g_neg * p_neg) / tf.reduce_sum(g_neg)

    # 3. FP
    fp = tf.reduce_sum(p_pos * g_neg) / tf.reduce_sum(g_neg)

    # 4. FN
    fn = tf.reduce_sum(p_neg * g_pos) / tf.reduce_sum(g_pos)

    # 5. Precision
    prec = tf.reduce_sum(g_pos * p_pos) / tf.reduce_sum(p_pos)

    if step >= 0:
        tf.summary.scalar(name + '/acc', acc, step=step)
        tf.summary.scalar(name + '/tp', tp, step=step)
        tf.summary.scalar(name + '/tn', tn, step=step)
        tf.summary.scalar(name + '/fp', fp, step=step)
        tf.summary.scalar(name + '/fn', fn, step=step)
        tf.summary.scalar(name + '/precision', prec, step=step)
        tf.summary.histogram(name + '/hist', pred, step=step)

    return acc, tp, tn, fp, fn, prec
