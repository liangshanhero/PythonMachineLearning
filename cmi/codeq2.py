import tensorflow as tf

a = tf.placeholder(shape=[3, 2], dtype=tf.float32)

b = a * 7

c = tf.nn.softmax(b)

sess = tf.Session()

print(sess.run(c, feed_dict={a: [[1, 2], [3, 4], [5, 6]]}))


