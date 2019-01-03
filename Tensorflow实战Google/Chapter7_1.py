import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_example(pixels, label, image):
    print(image)
    image_raw = image.tostring()
    print(image_raw)
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


mnist = input_data.read_data_sets('./Data/mnist', dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
print(images.shape)
num_examples = mnist.train.num_examples

with tf.python_io.TFRecordWriter("./Result/chapter7_1_output.tfrecords") as writer:
    for index in range(num_examples):
        example = _make_example(pixels,labels[index],images[index])
        writer.write(example.SerializeToString())

print("TFRecord training has been saved")

images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

with tf.python_io.TFRecordWriter("./Result/chapter7_1_output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(pixels_test,labels_test[index],images_test[index])
        writer.write(example.SerializeToString())

print("TFRecord test has been saved")