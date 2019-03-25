import tensorflow as tf

xData = [1, 2, 3, 4, 5]
yTrainData = [3, 5, 7, 9, 11]

x = tf.placeholder(shape=[], dtype=tf.float32)
yTrain = tf.placeholder(shape=[], dtype=tf.float32)

w = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

y = w * x + b

loss = tf.abs(yTrain - y)

optimizer = tf.train.GradientDescentOptimizer(0.004)

train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(1000000):
    lossSum = 0
    for j in range(3):
        result = sess.run([train, x, w, b, y, yTrain, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
        lossSum = lossSum + float(result[len(result) - 1])

    avgLoss = (lossSum / 3)
    print(result)
    print("i: %d, avgLoss: %10.10f" % (i, avgLoss))
    if avgLoss < 0.01:
        break
