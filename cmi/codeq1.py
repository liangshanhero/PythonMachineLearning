import tensorflow as tf

a = tf.placeholder(shape=[3, 2], dtype=tf.int32)

b = a * 7

sess = tf.Session()

print(sess.run(b, feed_dict={a: [[1, 2], [3, 4], [5, 6]]}))


