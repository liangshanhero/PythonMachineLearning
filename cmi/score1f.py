
#tensorfolw 1.10 OK, 1.13 not OK.
import tensorflow as tf
x=tf.placeholder(shape=[3], dtype=tf.float32)
yTrain=tf.placeholder(shape=[], dtype=tf.float32)
w=tf.Variable(tf.zeros([3]),dtype=tf.float32)

wn=tf.nn.softmax(w)

n=x*wn





y=tf.reduce_sum(n)
loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(0.1)


train=optimizer.minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(5000):
    result=sess.run([train, x, w,wn,y,yTrain, loss], feed_dict={x:[90,80,70],yTrain:85})
    print(result[3])
    result=sess.run([train, x, w,wn,y,yTrain, loss], feed_dict={x:[98,95,87],yTrain:96})

    print(result[3])
