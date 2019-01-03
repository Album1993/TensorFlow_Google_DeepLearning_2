import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

y1 = tf.constant([1.0, 2.0, 3.0])
y2 = tf.constant([3.0, 4.0, 5.0])

sess = tf.Session()
with sess.as_default():
    print(tf.reduce_mean(v).eval())

    # MSE
    mse = tf.reduce_mean(tf.square(y1 - y2))
    print(mse.eval())
