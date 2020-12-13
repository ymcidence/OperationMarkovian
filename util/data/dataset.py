from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
from meta import ROOT_PATH
from util.data.feat import *

AUTOTUNE = tf.data.experimental.AUTOTUNE


class BasicData(object):
    def __init__(self, file_name, batch_size, training=True):
        self.file_name = file_name
        self.batch_size = batch_size
        self.training = training
        self.meta = np.load(os.path.join(ROOT_PATH, 'data', 'ms_' + self.file_name + '.npy'), allow_pickle=True).item()
        # noinspection PyUnresolvedReferences
        self.max_time = self.meta['max_time']
        # noinspection PyUnresolvedReferences
        self.event_type = self.meta['event_type']
        self._load_data()

    def _get_map_function(self):
        def map_function(tf_example: tf.train.Example):
            feat_dict = {
                'event_feat': tf.io.FixedLenFeature([self.max_time * EVENT_FEAT.__len__()], tf.float32),
                'event_type': tf.io.FixedLenFeature([self.max_time], tf.float32),
                'event_time': tf.io.FixedLenFeature([self.max_time], tf.float32),
                'session_feat': tf.io.FixedLenFeature([self.max_time * SESSION_FEAT.__len__()], tf.float32),
                'user_feat': tf.io.FixedLenFeature([self.max_time * USER_FEAT.__len__()], tf.float32),
                'mask': tf.io.FixedLenFeature([self.max_time], tf.float32),
                'guid': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'seq_length': tf.io.FixedLenFeature([], tf.int64)
            }
            features = tf.io.parse_single_example(tf_example, features=feat_dict)

            features['event_feat'] = tf.reshape(features['event_feat'], [self.max_time, -1])
            features['event_type'] = tf.cast(features['event_type'], tf.int32)
            features['session_feat'] = tf.reshape(features['session_feat'], [self.max_time, -1])
            features['user_feat'] = tf.reshape(features['user_feat'], [self.max_time, -1])
            features['label'] = tf.cast(features['label'], tf.float32)
            return features

        def filter_pos(inputs):
            return tf.equal(inputs['label'], 1)

        def filter_neg(inputs):
            return tf.equal(inputs['label'], 0)

        return map_function, filter_pos, filter_neg

    def _load_data(self):
        map_function, filter_pos, filter_neg = self._get_map_function()


        def read_record(file_name):
            return tf.data.TFRecordDataset(file_name) \
                .map(map_function, num_parallel_calls=AUTOTUNE) \
                .prefetch(AUTOTUNE)

        train_record = os.path.join(ROOT_PATH, 'data', 'train_' + self.file_name + '.tfrecords')
        test_record = os.path.join(ROOT_PATH, 'data', 'test_' + self.file_name + '.tfrecords')

        train_data = read_record(train_record)
        test_data = read_record(test_record)

        pos_batch_size = int(self.batch_size * .1)
        neg_batch_size = int(self.batch_size - pos_batch_size)
        # noinspection PyUnresolvedReferences
        self.pos_data = iter(train_data.filter(filter_pos).cache().repeat().shuffle(self.meta['train_pos_size']).batch(
            pos_batch_size))
        # noinspection PyUnresolvedReferences
        self.neg_data = iter(train_data.filter(filter_neg).cache().repeat().shuffle(self.meta['train_neg_size']).batch(
            neg_batch_size))

        if self.training:
            # noinspection PyUnresolvedReferences
            self.test_data = iter(test_data.cache().repeat().shuffle(self.meta['test_size']).batch(self.batch_size))
        else:
            self.test_data = iter(test_data.cache().batch(1))

    def next(self, training_batch=True):
        if training_batch:
            pos = next(self.pos_data)
            neg = next(self.neg_data)
            rslt = dict()
            for k in pos.keys():
                rslt[k] = tf.concat([pos[k], neg[k]], axis=0)

        else:
            rslt = next(self.test_data)

        return rslt


if __name__ == '__main__':
    ds = BasicData('feat', 20)
    a = ds.next()
    print(a)
