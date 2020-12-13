from __future__ import print_function, absolute_import, division, unicode_literals

import tensorflow as tf
from layer.transformer.isab import MultiheadAttentionBlock as MAB
from layer.input_layer import InputLayer
from layer.transformer.attention import create_padding_mask


class BasicMAB(tf.keras.Model):

    def __init__(self, n_head, d_model, max_time, dff, event_type=13, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.max_time = max_time
        self.n_head = n_head
        self.event_type = event_type
        self.dff = dff
        self.input_layer = InputLayer(d_model, max_time, event_type)
        self.mab = MAB(d_model, n_head, dff=dff)
        self.inducer = tf.Variable(initial_value=tf.random.normal([d_model], stddev=.01), trainable=True,
                                   dtype=tf.float32, name='inducer')
        self.cls = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        event_feat = inputs['event_feat']
        _mask = inputs['mask']  # [N T]

        mask = create_padding_mask(_mask, tar=0)

        batch_size = tf.shape(event_feat)[0]
        inducer = tf.expand_dims(tf.expand_dims(self.inducer, axis=0), axis=0)  # [1 1 D]
        inducer = tf.tile(inducer, [batch_size, 1, 1])  # [N 1 D]

        input_feat = self.input_layer(inputs)  # [N T D]

        hidden, att = self.mab(inducer, input_feat, training, mask=mask)  # [N 1 D] [N 1 T]

        cls = self.cls(tf.squeeze(hidden, axis=1))

        return cls, att

    def get_config(self):
        pass

    @staticmethod
    def obj(cls, label, step=-1):
        bce = tf.keras.losses.BinaryCrossentropy()
        l = label[:, tf.newaxis]
        loss = tf.reduce_mean(bce(l, cls))

        if step >= 0:
            tf.summary.scalar('loss/bce', loss, step=step)

        return loss


if __name__ == '__main__':
    from util.data.dataset import BasicData

    data = BasicData('feat', 20)
    model = BasicMAB(4, 128, 100, 128)
    x = data.next()
    a, b = model(x)

    print(a)
