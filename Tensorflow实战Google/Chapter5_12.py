import tensorflow as tf

saver = tf.train.import_meta_graph("./result/Chapter5_10.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./result/Chapter5_10.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
