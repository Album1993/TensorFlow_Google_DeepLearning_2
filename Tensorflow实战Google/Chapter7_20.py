import tensorflow as tf
import numpy as np

train_files = tf.train.match_filenames_once("./Result/chapter7_1_output.tfrecords")
test_files = tf.train.match_filenames_once("./Result/chapter7_1_output_test.tfrecords")


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)

    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    return images, labels


image_size = 299
batch_size = 100
shuffle_buffer = 10000

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 50000000000

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch,weights1,biases1,weights2,biases2)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
regularization = regularizer(weights1) + regularizer(weights2)

loss = cross_entropy_mean + regularization

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)


test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser)
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_image_batch,test_label_batch = test_iterator.get_next()

test_logit = inference(test_image_batch, weights1,biases1,weights2,biases2)
predictions = tf.argmax(test_logit,axis=-1,output_type=tf.int32)

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))

    sess.run(iterator.initializer)

    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break

    test_results = []
    test_labels = []

    sess.run(test_iterator.initializer)

    while True:
        try:
            pred,label = sess.run([predictions,test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

correct=[float(y==y_) for (y,y_) in zip(test_results,test_labels)]
accuracy = sum(correct) / len(correct)
print("Test accuracy is:", accuracy)
