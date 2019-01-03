import tensorflow as tf

sess = tf.Session()
with sess.as_default():
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(tf.log(v).eval())
