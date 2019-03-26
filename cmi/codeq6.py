import tensorflow as tf
import numpy as np

m1 = tf.placeholder(shape=[3, 3], dtype=tf.float32)
m2 = tf.placeholder(shape=[3, 4], dtype=tf.float32)

y = tf.matmul(m1, m2)

sess = tf.Session()

print(sess.run([y], feed_dict={m1: [[1, 2, 3], [7, 8, 9], [4, 5, 6]], m2: [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]}))

print(np.matmul([[1, 2, 3], [7, 8, 9], [4, 5, 6]], [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]))


