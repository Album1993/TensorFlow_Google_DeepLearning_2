import tensorflow as tf
saver = tf.train.import_meta_graph("./Result/Chapter5_7.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess,"./Result/Chapter5_7.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))