import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100).astype(np.float32)
y_Label = x * 3 + 8

plt.scatter(x, y_Label, marker='+')
plt.show()

# exit(0)
# print(x, y_Label)

# tf.Variable是tf的变量，是会被训练优化的，非tf.Variable值不会变
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = x * Weights + biases

loss = tf.reduce_mean(tf.square(y - y_Label))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(201):
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run(loss), sess.run(Weights), sess.run(biases))

