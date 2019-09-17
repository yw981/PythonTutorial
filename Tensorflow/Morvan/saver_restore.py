import tensorflow as tf
import numpy as np

# restore variables
# redefine the same shape and same type for your variables
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32)
W = tf.Variable(np.ones((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32)
b = tf.Variable(np.ones((2, 3)), dtype=tf.float32, name="biases")
r = W + b

# not need init step

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("weights origin", sess.run(W))
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
    print("r:", sess.run(r))
