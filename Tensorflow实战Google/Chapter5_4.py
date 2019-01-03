import tensorflow as tf

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
