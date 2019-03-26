import tensorflow as tf

x1 = tf.placeholder(shape=[], dtype=tf.float32)

x2 = tf.placeholder(shape=(), dtype=tf.float32)

x3 = tf.placeholder(shape=[3], dtype=tf.float32)

x4 = tf.placeholder(shape=(3), dtype=tf.float32)

x5 = tf.placeholder(shape=3, dtype=tf.float32)

x6 = tf.placeholder(shape=(3, ), dtype=tf.float32)

x7 = tf.placeholder(shape=(2, 3), dtype=tf.float32)

x8 = tf.placeholder(shape=[2, 3], dtype=tf.float32)

sess = tf.Session()

result = sess.run(tf.shape(x1), feed_dict={x1: 8})
print(result)

result = sess.run(tf.shape(x2), feed_dict={x2: 8})
print(result)

result = sess.run(tf.shape(x3), feed_dict={x3: [1, 2, 3]})
print(result)

result = sess.run(tf.shape(x4), feed_dict={x4: [1, 2, 3]})
print(result)

result = sess.run(tf.shape(x5), feed_dict={x5: [1, 2, 3]})
print(result)

result = sess.run(tf.shape(x6), feed_dict={x6: [1, 2, 3]})
print(result)

result = sess.run(tf.shape(x7), feed_dict={x7: [[1, 2, 3], [2, 4, 6]]})
print(result)

result = sess.run(tf.shape(x8), feed_dict={x8: [[1, 2, 3], [2, 4, 6]]})
print(result)
