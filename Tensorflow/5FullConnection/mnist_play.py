# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def decode(datum):
    return np.argmax(datum)


def show_images(mnist, start, width, height):
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    for i in range(height):
        for j in range(width):
            idx = start + (i * width) + j
            img = mnist.train.images[idx].reshape([28, 28])
            posp = fig.add_subplot(height, width, (i * width) + j + 1)
            plt.title(str(idx) + ':' + str(decode(mnist.train.labels[idx])))
            posp.imshow(img, cmap=plt.cm.gray)

    plt.show()


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    show_images(mnist, 0, 10, 4)

