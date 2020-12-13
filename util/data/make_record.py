from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from meta import ROOT_PATH
from util.data.feat import *
from util.processing import get_mean_std


def _int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _bytes_feature(value):
    """Create a feature that is stored on disk as a byte array."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert(writer: tf.io.TFRecordWriter, event_feat, event_type, event_time, session_feat, user_feat, mask, guid,
             label,
             max_time):
    features = tf.train.Example(
        features=tf.train.Features(feature={
            "event_feat": _float_feature(event_feat),
            "event_type": _float_feature(event_type),
            "event_time": _float_feature(np.asarray(event_time, np.float)),
            "session_feat": _float_feature(session_feat),
            "user_feat": _float_feature(user_feat),
            "mask": _float_feature(mask),
            "guid": _bytes_feature(guid.encode('utf-8')),
            "label": _int64_feature(label),
            "seq_length": _int64_feature(max_time)
        })
    )
    writer.write(features.SerializeToString())


class DataMaker(object):
    def __init__(self, file_name, max_time, event_type=13, norm_file=None):
        self.file_name = file_name
        self.max_time = max_time
        self.event_type = event_type
        self.norm_file = norm_file
        self.test_proportion = .1

        file_name = os.path.join(ROOT_PATH, 'data', self.file_name)
        self.raw_data: pd.DataFrame = pq.read_table(file_name).to_pandas()

        sess_time = self.raw_data['sessionStartTime']
        self.event_size = self.raw_data.shape[0]
        self.raw_data['sessionStartTime1'] = [sess_time[i].hour for i in range(self.event_size)]

        self.guid = np.asarray(self.raw_data['guid'].drop_duplicates().values)
        
        self.label = self._get_label() #np.asarray([self.raw_data[self.raw_data['guid'] == g]['flag'].values[0] for g in self.guid])
        self.total_size = self.guid.shape[0]
        self.test_size = int(np.floor(self.total_size * self.test_proportion))
        self.training_size = self.total_size - self.test_size

        self.mean, self.std = get_mean_std(self.raw_data[FEAT_LIST], FEAT_LIST, NORM_LIST)
        self._split()
    def _get_label(self):
        rslt = np.zeros(self.guid.shape[0])
        for i, g in enumerate(self.guid):
            if i % 1000 == 0:
                print(i, g)
            rslt[] = self.raw_data[self.raw_data['guid']==g]['flag'].values[0])
        return np.asarray(rslt)

    def _split(self):
        np.random.seed(2)
        test_ind = np.random.choice(self.total_size, self.test_size, replace=False)
        train_ind = [i for i in range(self.total_size)]
        train_ind = np.asarray(list(set(train_ind) - set(test_ind)), dtype=np.int)
        self.train_ind = train_ind
        self.test_ind = test_ind

    def __call__(self):
        map_function = self._get_map_function()

        train_file_name = 'train_' + str(self.file_name).replace('.parquet', '.tfrecords')
        train_file_name = os.path.join(ROOT_PATH, 'data', train_file_name)
        train_writer = tf.io.TFRecordWriter(train_file_name)
        test_file_name = 'test_' + str(self.file_name).replace('.parquet', '.tfrecords')
        test_file_name = os.path.join(ROOT_PATH, 'data', test_file_name)
        test_writer = tf.io.TFRecordWriter(test_file_name)

        for n, i in enumerate(self.train_ind):
            if n % 1000 == 0:
                print(n)
            this_guid = self.guid[i]
            this_label = self.label[i]
            event_feat, event_type, event_time, session_feat, user_feat, mask, guid, label = map_function(this_guid,
                                                                                                          this_label)

            _convert(train_writer, event_feat, event_type, event_time, session_feat, user_feat, mask, guid, label,
                     self.max_time)

        for n, i in self.test_ind:
            if n % 1000 == 0:
                print('_' + str(n))
            this_guid = self.guid[i]
            this_label = self.label[i]
            event_feat, event_type, event_time, session_feat, user_feat, mask, guid, label = map_function(this_guid,
                                                                                                          this_label)

            _convert(test_writer, event_feat, event_type, event_time, session_feat, user_feat, mask, guid, label,
                     self.max_time)

        train_writer.close()
        test_writer.close()

        ms_name = os.path.join(ROOT_PATH, 'data', 'ms_' + str(self.file_name).replace('.parquet', '.npy'))
        ms = {'mean': self.mean,
              'std': self.std,
              'test_size': self.test_ind.__len__(),
              'train_pos_size': np.where(self.label[self.train_ind] == 1)[0].__len__(),
              'train_neg_size': np.where(self.label[self.train_ind] == 0)[0].__len__(),
              'max_time': self.max_time,
              'event_type': self.event_type}
        np.save(ms_name, ms)

    def _get_map_function(self):
        def _map_function(guid, label):
            df: pd.DataFrame = self.raw_data[self.raw_data['guid'] == guid].sort_values('eventTime')

            user_feat = self._process_data(df, USER_FEAT)[0]
            session_feat = self._process_data(df, SESSION_FEAT)[0]
            event_feat, mask = self._process_data(df, EVENT_FEAT)

            event_type = self._trim_or_padding(np.asarray(df['eventType'].values, np.int32))[0]
            event_time = self._process_time(df)[0]

            return event_feat, event_type, event_time, session_feat, user_feat, mask, guid, label

        return _map_function

    def _process_data(self, raw_data: pd.DataFrame, feat_part):
        values = np.asarray(raw_data[feat_part].values, np.float32)
        norm_ind = [FEAT_LIST.index(i) for i in feat_part]
        std = self.std[norm_ind]
        mean = self.mean[norm_ind]
        rslt = self._trim_or_padding((values - mean) / std)
        return rslt

    def _process_time(self, raw_data: pd.DataFrame):
        e = raw_data['eventTime'].values
        start = e[0]
        return self._trim_or_padding(np.asarray([(i - start) / np.timedelta64(1, 's') for i in e]))

    def _trim_or_padding(self, x):
        rslt = x
        shape = rslt.shape
        if shape[0] > self.max_time:
            if shape.__len__() > 1:
                rslt = rslt[:self.max_time, :]
            else:
                rslt = rslt[:self.max_time]
            mask = np.ones(self.max_time)
        elif shape[0] < self.max_time:
            d = self.max_time - shape[0]
            padding_shape = d if shape.__len__() <= 1 else [d, shape[1]]
            padding = np.zeros(padding_shape, dtype=rslt.dtype)
            rslt = np.concatenate([rslt, padding], axis=0)

            mask_1 = np.ones(shape[0])
            mask_0 = np.zeros(d)
            mask = np.concatenate([mask_1, mask_0], axis=0)

        else:
            mask = np.ones(self.max_time)

        return rslt, mask


if __name__ == '__main__':
    maker = DataMaker('feat.parquet', 100)
    maker()
