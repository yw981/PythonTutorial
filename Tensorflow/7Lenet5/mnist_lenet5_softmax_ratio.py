# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np

TEST_INTERVAL_SECS = 5


def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
        y = mnist_lenet5_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 忽略第二列
        y_t2, indices = tf.math.top_k(y, 2)

        y_pred = tf.reduce_max(y, 1)
        y_pred_mean = tf.reduce_mean(y_pred)
        y_pred_right = tf.boolean_mask(y_pred, correct_prediction)
        y_pred_right2 = tf.boolean_mask(y_t2, correct_prediction)

        y_pred_wrong = tf.boolean_mask(y_pred, ~correct_prediction)
        y_pred_wrong2 = tf.boolean_mask(y_t2, ~correct_prediction)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images, (
                        mnist.test.num_examples,
                        mnist_lenet5_forward.IMAGE_SIZE,
                        mnist_lenet5_forward.IMAGE_SIZE,
                        mnist_lenet5_forward.NUM_CHANNELS))
                    accuracy_score, vy_pred_mean, vy_pred_right, vy_pred_wrong, vy_t2, vy_pred_right2, vy_pred_wrong2 = sess.run(
                        [accuracy, y_pred_mean, y_pred_right, y_pred_wrong, y_t2, y_pred_right2, y_pred_wrong2],
                        feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print("Softmax pred mean (all) = %g" % vy_pred_mean)
                    np.savetxt('vy_pred_right2.npy', vy_pred_right2)

                    np.savetxt('vy_pred_wrong2.npy', vy_pred_wrong2)
                    # print(vy_t2)
                    np.savetxt('vy_t2.npy', vy_t2)
                    print("right , wrong = %g , %g" % (np.mean(vy_pred_right), np.mean(vy_pred_wrong)))
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    c1, c2 = np.split(vy_pred_right2, 2, axis=1)
                    ratio_r = c1 / c2
                    np.savetxt('ratio_right.npy', ratio_r)
                    print('ratio_right min %g max %g mean %g var %g ' %
                          (np.min(ratio_r), np.max(ratio_r), np.mean(ratio_r), np.var(ratio_r)))

                    w2 = np.loadtxt('vy_pred_wrong2.npy')
                    c1, c2 = np.split(vy_pred_wrong2, 2, axis=1)
                    ratio_w = c1 / c2
                    np.savetxt('ratio_wrong.npy', ratio_w)
                    print('ratio_wrong min %g max %g mean %g var %g  ' %
                          (np.min(ratio_w), np.max(ratio_w), np.mean(ratio_w), np.var(ratio_w)))
                    return
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
