import tensorflow as tf
import random

a = tf.placeholder(dtype=tf.float32)

b = tf.nn.sigmoid(a)

sess = tf.Session()

for i in range(5):
    print(sess.run(b, feed_dict={a: random.random() * 40 - 20}))


