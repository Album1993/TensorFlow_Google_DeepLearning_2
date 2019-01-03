import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0])
v2 = tf.constant([3.0, 4.0, 5.0])

sess = tf.Session()

a = tf.constant(2.0)
b = tf.constant(3.0)
with sess.as_default():
    loss = tf.reduce_sum(tf.where(tf.greater(v1, v2), (v1 - v2) * 2, (v2 - v1) * 3))
    print(loss.eval())
    loss2 = tf.reduce_sum(tf.where(tf.greater(v1, v2), (v1 - v2) * a, (v2 - v1) * b))
    print(loss2.eval())
