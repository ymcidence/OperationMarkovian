from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model, max_time):
    # angle_rates = 1 / np.power(max_time, (2 * (i // 2)) / np.float32(d_model))

    angle_rates = 1 / tf.pow(max_time * 1., (2 * (i // 2)) / (1. * d_model))
    return pos * angle_rates * np.pi * 2


def sinusoid(position, d_model, max_time):
    angle_rads = get_angles(position[:, :, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model,
                            max_time)

    sin = tf.sin(angle_rads[..., 0::2])

    cos = tf.cos(angle_rads[..., 1::2])

    rslt = tf.concat([tf.expand_dims(sin, 3), tf.expand_dims(cos, 3)], axis=3)

    batch_size = tf.shape(angle_rads)[0]

    rslt = tf.reshape(rslt, [batch_size, -1, d_model, 1])

    rslt = tf.squeeze(rslt, axis=3)

    return rslt


class InputLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, max_time, event_type=13, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_time = max_time
        self.event_type = event_type
        self.event_emb = tf.keras.layers.Embedding(event_type + 1, d_model)
        self.event_fc = tf.keras.layers.Dense(d_model)
        self.session_fc = tf.keras.layers.Dense(d_model)
        self.user_fc = tf.keras.layers.Dense(d_model)

    def call(self, inputs, **kwargs):
        event_feat = inputs['event_feat']
        event_type = inputs['event_type']
        time_stamp = inputs['event_time']
        session_feat = inputs['session_feat']
        user_feat = inputs['user_feat']

        event = self._event_callback(event_feat, event_type, time_stamp)
        session = self._session_callback(session_feat) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        user = self._user_callback(user_feat) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        return event + session + user

    def _event_callback(self, event_feat, event_type, time_stamp):
        """

        :param event_feat: [N T D]
        :param event_type: [N T]
        :param time_stamp: [N T]
        :return:
        """
        x = self.event_fc(event_feat) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.event_emb(event_type) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += sinusoid(time_stamp, self.d_model, self.max_time)

        return x

    def _session_callback(self, session_feat):
        return self.session_fc(session_feat)

    def _user_callback(self, user_feat):
        return self.user_fc(user_feat)


if __name__ == '__main__':
    p = np.asarray([[2, 3, 4, 5], [1, 1, 1, 1]])
    d = 32
    m = 10
    r = sinusoid(p, d, m)
    print(r)
