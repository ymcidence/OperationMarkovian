from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
from model.basic_mab import BasicMAB
from util.data.dataset import BasicData
from run.config import parser
from util.processing import make_dir
from util.eval import hook_cls


def step_train(model: BasicMAB, data: BasicData, opt: tf.optimizers.Optimizer, t):
    x = data.next()
    step = t if t % 10 == 0 else -1
    with tf.GradientTape() as tape:
        cls, att = model(x)
        loss = model.obj(cls, x['label'])
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

        if t % 10 == 0:
            img = tf.squeeze(att, 2)[0, :, :100]
            tf.summary.scalar('train/loss', loss, step=t)
            tf.summary.image('train/att', img[tf.newaxis, :, :, tf.newaxis], step=t)

        step_test(model, data, t)

    return loss.numpy()


def step_test(model: BasicMAB, data: BasicData, t):
    if t % 10 != 0:
        return
    x = data.next(False)

    cls, att = model(x, training=False)
    hook_cls(tf.squeeze(cls), tf.squeeze(x['label']), 'test', t)


def main():
    args = parser.parse_args()
    data = BasicData(args.file_name, args.batch_size)
    model = BasicMAB(args.n_head, args.d_model, args.max_sec, args.dff, data.event_type)
    opt = tf.keras.optimizers.Adam(args.learning_rate)

    summary_path, save_path = make_dir(args.task_name)
    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(opt=opt, model=model)

    for i in range(args.max_iter):
        with writer.as_default():
            nll = step_train(model, data, opt, i)
            print('Step: {}, Loss: {}'.format(i, nll))
            if i == 0:
                print(model.summary())

            if (i + 1) % 200 == 0:
                save_name = os.path.join(save_path, 'ym' + str(i))
                checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    main()
