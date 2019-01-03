import tensorflow as tf

def parser(record):
    ...


input_files = tf.placeholder(tf.string)

dataset = tf.data.TFRecordDataset(input_files)

dataset = dataset.map(parser)

iterator = dataset.make_initializable_iterator()

feat1, feat2 = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer,feed_dict={input_files:[]})

    while True:
        try:
            sess.run([feat1,feat2])
        except tf.errors.OutOfRangeError:
            break

