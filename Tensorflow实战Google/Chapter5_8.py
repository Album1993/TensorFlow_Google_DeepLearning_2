import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(4.0, shape=[1]), name="v2")


result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"./Result/Chapter5_7.ckpt")
    print(sess.run(result))
