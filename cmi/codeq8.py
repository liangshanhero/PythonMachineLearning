import tensorflow as tf
import numpy as np
import sys

predictData = None

argt = sys.argv[1:]

for v in argt:
    if v.startswith("-predict="):
        tmpStr = v[len("-predict="):]
        predictData = np.fromstring(tmpStr, dtype=np.float32, sep=",")

print("predictData: %s" % predictData)

x1 = tf.placeholder(dtype=tf.float32)
x2 = tf.placeholder(dtype=tf.float32)
x3 = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(0.1, dtype=tf.float32)
w2 = tf.Variable(0.1, dtype=tf.float32)
w3 = tf.Variable(0.1, dtype=tf.float32)

n1 = x1 * w1
n2 = x2 * w2
n3 = x3 * w3

y = n1 + n2 + n3

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.001)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(200):
	result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 90, x2: 80, x3: 70, yTrain: 85})
	print(result)

	result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss], feed_dict={x1: 98, x2: 95, x3: 87, yTrain: 96})
	print(result)

if predictData is not None:
    print(sess.run(y, feed_dict={x1: predictData[0], x2: predictData[1], x3: predictData[2]}))
