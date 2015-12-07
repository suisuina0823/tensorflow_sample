# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import input_data
import tensorflow as tf

print "****read MNIST data****"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print "****Start Tutorial****"
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# In this case, we ask TensorFlow to minimize cross_entropy
# using the gradient descent algorithm with a learning rate of 0.01.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

print "****init****"
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print "****tarin 1000times****"
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# result
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})