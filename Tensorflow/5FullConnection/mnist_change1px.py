# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import numpy as np
import matplotlib.pyplot as plt

TEST_INTERVAL_SECS = 5

def show_images(data, width, height):
    fig = plt.figure()
    # plt.subplots_adjust(wspace=1, hspace=1)
    idx = 0
    for i in range(height):
        for j in range(width):
            img = data[idx].reshape([28, 28])
            posp = fig.add_subplot(height, width, (i * width) + j + 1)
            # plt.title(str(idx) + ':' + str(decode(mnist.train.labels[idx])))
            posp.imshow(img, cmap=plt.cm.gray)
            idx += 1

    plt.show()

# 原始 test accuracy = 0.9783
# 必须先运行mnist_backward训练

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    image_num = 1000
                    test_images = mnist.test.images[0:image_num,:]
                    test_labels = mnist.test.labels[0:image_num]

                    # test_image = mnist.test.images[0:1,:]
                    # test_label = mnist.test.labels[0:1]
                    # print(test_label)

                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # print(np.array(test_image).shape)
                    # print(np.array(test_label).shape)

                    generate_images = []
                    generate_labels = []
                    for k in range(0,image_num):
                        for i in range(0, 784):
                            # 传值，默认传引用
                            img = np.squeeze(test_images[k])
                            tmp = img[i]
                            img[i] = 1
                            generate_images.append(np.copy(img))
                            generate_labels.append(np.squeeze(test_labels[k]))
                            img[i] = tmp

                    print(np.array(generate_images).shape)
                    print(np.array(generate_labels).shape)
                    # print(generate_images)
                    # show_images(generate_images,8,8)

                    # pred_result = sess.run(correct_prediction, feed_dict={x: generate_images, y_: generate_labels})
                    # print("Result ", pred_result)

                    accuracy_score = sess.run(accuracy, feed_dict={x: generate_images, y_: generate_labels})

                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
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


# # import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
#
# def decode(datum):
#     return np.argmax(datum)
#
#
# def show_images(mnist, start, width, height):
#     fig = plt.figure()
#     plt.subplots_adjust(wspace=1, hspace=1)
#     for i in range(height):
#         for j in range(width):
#             idx = start + (i * width) + j
#             img = mnist.train.images[idx].reshape([28, 28])
#             posp = fig.add_subplot(height, width, (i * width) + j + 1)
#             plt.title(str(idx) + ':' + str(decode(mnist.train.labels[idx])))
#             posp.imshow(img, cmap=plt.cm.gray)
#
#     plt.show()
#
#
# if __name__ == '__main__':
#     mnist = input_data.read_data_sets("../data/", one_hot=True)
#     # show_images(mnist, 0, 10, 4)
#     img = mnist.train.images[0].reshape([28, 28])
#     img[0][0] = 1
#     print(img)
#     fig = plt.figure()
#     plt.imshow(img)
#     plt.show()
#
