import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels
import os

# img_path = input('Input the path and image name:')
img_path = ['pic/0.jpg', 'pic/1.jpg']
ready_images = []
path = 'pic/'
for filename in os.listdir(path):
    file_path = os.path.join(path,filename)

    img_ready = utils.load_image(file_path)
    print(file_path)
    img_ready = img_ready.reshape((224, 224, 3))
    # print(img_ready.shape)
    ready_images.append(img_ready)

ready_images = np.array(ready_images)
print(ready_images.shape)

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, ready_images.shape)
    vgg = vgg16.Vgg16()
    vgg.forward(images)
    probability = sess.run(vgg.prob, feed_dict={images: ready_images})
    print(probability.shape)
    # print(probability)
    result_index = np.argmax(probability,axis=1)
    print(result_index.shape)
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)
    for i,label_index in enumerate(result_index):
        print('# '+str(i)+' '+labels[label_index])

