import tensorflow as tf

x = tf.placeholder(shape=[1, 3], dtype=tf.float32, name="x")

w = tf.Variable(tf.ones([3, 3]), dtype=tf.float32, name="w")
b = tf.Variable(1, dtype=tf.float32, name="b")

y = tf.add(tf.matmul(x, w, name="MatMul"), b, name="y-Add")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(y, feed_dict={x: [[1, 2, 3]]})

writer = tf.summary.FileWriter("graph", sess.graph)



