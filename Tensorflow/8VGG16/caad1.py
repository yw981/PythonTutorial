import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels
import os

ready_images = []
path = 'C:/PROJECT/Python/images_round1/dataset0/'
k = 0
filenames = os.listdir(path)
for filename in filenames:
    file_path = os.path.join(path, filename)

    img_ready = utils.load_image(file_path)
    # print(file_path)

    img_ready = img_ready.reshape((224, 224, 3))
    # print(img_ready.shape)
    ready_images.append(img_ready)
    k += 1
    if k > 10:
        break

ready_images = np.array(ready_images)
print(ready_images.shape)

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, ready_images.shape)
    vgg = vgg16.Vgg16()
    vgg.forward(images)
    probability = sess.run(vgg.prob, feed_dict={images: ready_images})
    print(probability.shape)
    # print(probability)
    result_index = np.argmax(probability, axis=1)
    print(result_index.shape)
    # top5 = np.argsort(probability[0])[-1:-6:-1]
    # print("top5:", top5)
    for i, label_index in enumerate(result_index):
        print(str(i) + ',' + str(filenames[i]) + ',' + str(label_index) + ',' + labels[label_index])
