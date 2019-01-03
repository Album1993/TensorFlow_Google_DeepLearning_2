import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = tf.add(a, b, name="add")
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess1.run(result)
sess1.close()

sess2 = tf.Session(config=config)
sess2.run(result)
sess2.close()
