import tensorflow as tf

sess = tf.Session()
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = tf.add(a, b, name="add")
with sess.as_default():
    print(result.eval())
    print(sess.run(result))
    print(result.eval(session=sess))
