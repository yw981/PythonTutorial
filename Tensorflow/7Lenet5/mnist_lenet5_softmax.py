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

        y_pred = tf.reduce_max(y, 1)
        y_pred_mean = tf.reduce_mean(y_pred)
        y_pred_right = tf.boolean_mask(y_pred, correct_prediction)
        # y_pred_right_mean = tf.reduce_mean(y_pred_right)

        y_pred_wrong = tf.boolean_mask(y_pred, ~correct_prediction)
        # y_pred_wrong_mean = tf.reduce_mean(y_pred_wrong)



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
                    accuracy_score, vy_pred_mean, vy_pred_right, vy_pred_wrong = sess.run(
                        [accuracy, y_pred_mean, y_pred_right, y_pred_wrong],
                        feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print("Softmax pred mean (all) = %g" % vy_pred_mean)
                    np.savetxt('vy_pred_right', vy_pred_right)
                    np.savetxt('vy_pred_wrong', vy_pred_wrong)
                    print("right , wrong = %g , %g" % (np.mean(vy_pred_right), np.mean(vy_pred_wrong)))
                    # with open('result.txt', 'w') as fw:
                    #     fw.write(str(y_re.tostring()))
                    #     fw.write('\n')
                    #     fw.write(str(correct_prediction.tostring()))
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
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
