import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(
    tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less)) + tf.contrib.layers.l2_regularizer(0.5)(w1)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 50000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(learning_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    print(sess.run(w1))
