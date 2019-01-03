import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32,name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"./result/Chapter5_13.ckpt")
    print(sess.run(v))
