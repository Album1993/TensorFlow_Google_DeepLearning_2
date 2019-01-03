import tensorflow as tf

sess = tf.Session()

v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])

with sess.as_default():
    print((v1 * v2).eval())
    print(tf.matmul(v1, v2).eval())
