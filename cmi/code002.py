import tensorflow as tf

x = tf.placeholder(shape=[2, 3], dtype=tf.float32)

xShape = tf.shape(x)

sess = tf.Session()

result = sess.run(xShape, feed_dict={x: [[1, 2, 3], [2, 4, 6]]})
print(result)

result = sess.run(xShape, feed_dict={x: [1, 2, 3]})
print(result)

