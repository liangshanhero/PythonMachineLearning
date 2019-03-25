import tensorflow as tf
import sys


learnRateT = 0.001

argt = sys.argv[1:]
print("argt: %s" % argt)

for v in argt:
    if v.startswith("-learnrate="):
        learnRateT = float(v[11:])

xData = [1, 2, 3, 4, 5]
yTrainData = [2, 5, 10, 17, 26]

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(tf.ones([1, 8]), dtype=tf.float32)
b1 = tf.Variable(0, dtype=tf.float32)

n1 = tf.nn.tanh(tf.matmul(tf.reshape(x, [1, 1]), w1) + b1)

w2 = tf.Variable(tf.ones([8, 1]), dtype=tf.float32)
b2 = tf.Variable(0, dtype=tf.float32)

n2 = tf.matmul(n1, w2) + b2

y = tf.reduce_sum(n2)

loss = tf.abs(yTrain - y)

optimizer = tf.train.GradientDescentOptimizer(learnRateT)

train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(1000000):
    lossSum = 0
    for j in range(3):
        result = sess.run([train, x, w1, b1, y, yTrain, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
        lossSum = lossSum + float(result[len(result) - 1])

    avgLoss = (lossSum / 3)
    print(result)
    print("i: %d, avgLoss: %10.10f" % (i, avgLoss))
    if avgLoss < 0.01:
        break
