# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import numpy as np

import matplotlib.pyplot as plt

TEST_INTERVAL_SECS = 5


# 必须先运行mnist_backward训练

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)
        # add softmax
        y = tf.nn.softmax(y)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 排序
        y_sort = tf.contrib.framework.sort(y, 1, 'DESCENDING')

        std = tf.constant([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        y_err = tf.reduce_sum(tf.math.square(y_sort - std), 1)

        y_err_right = tf.boolean_mask(y_err, correct_prediction)
        y_err_wrong = tf.boolean_mask(y_err, ~correct_prediction)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # print(mnist.test.images)
                    print(np.array(mnist.test.images).shape)
                    # \ #

                    accuracy_score, vy_sort, vy_err, vy_err_right, vy_err_wrong, vcr = sess.run(
                        [accuracy, y_sort, y_err, y_err_right, y_err_wrong, correct_prediction],
                        feed_dict={x: mnist.test.images,
                                   y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    print(vcr.shape)

                    np.savetxt('vcr.npy', vcr)
                    np.savetxt('vy_sort.npy', vy_sort)
                    np.savetxt('vy_err.npy', vy_err)
                    np.savetxt('vy_err_right.npy', vy_err_right)
                    np.savetxt('vy_err_wrong.npy', vy_err_wrong)

                    print('error_right min %g max %g mean %g var %g ' %
                          (np.min(vy_err_right), np.max(vy_err_right), np.mean(vy_err_right),
                           np.square(np.var(vy_err_right))))
                    print('error_wrong min %g max %g mean %g var %g ' %
                          (np.min(vy_err_wrong), np.max(vy_err_wrong), np.mean(vy_err_wrong),
                           np.square(np.var(vy_err_wrong))))

                    bins = np.linspace(0, 0.8, 100)
                    plt.figure()
                    plt.hist(vy_err_right, bins)
                    plt.figure()
                    plt.hist(vy_err_wrong, bins, color='orange')

                    bins1 = np.linspace(0, 0.04, 100)
                    plt.figure()
                    plt.hist(vy_err_right, bins1, color='green')
                    plt.figure()
                    plt.hist(vy_err_wrong, bins1, color='yellow')
                    plt.show()


                else:
                    print('No checkpoint file found')
                    return
            return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
