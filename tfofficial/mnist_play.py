import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(type(x_train))

print(x_train.shape)

thr = 16 / 256

org = x_train[1:5]
dst = x_train[3:7]


def show_images(images):
    width = height = int(math.ceil(math.sqrt(images.shape[0])))
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    for i in range(height):
        for j in range(width):
            idx = (i * width) + j
            if idx >= images.shape[0]:
                plt.show()
                return
            img = images[idx].reshape([28, 28])
            posp = fig.add_subplot(height, width, (i * width) + j + 1)
            posp.imshow(img)

    plt.show()


def generate_attack(from_image, to_image, epsilon=16 / 256):
    dis = to_image - from_image
    generate_image = from_image
    ops = np.fabs(dis) < epsilon
    generate_image[ops] = to_image[ops]
    ops = np.fabs(dis) >= epsilon
    generate_image[ops] += np.sign(to_image[ops]) * epsilon
    return generate_image


gen = generate_attack(org, dst, thr)
print(gen.shape)

show_images(gen)

# fig = plt.figure()

# plt.imshow(gen[1])
# plt.show()
