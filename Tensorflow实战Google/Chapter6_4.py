import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import Chapter6_2
import Chapter6_3

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        xs = mnist.validation.images

        x = tf.placeholder(tf.float32, [mnist.validation.num_examples, Chapter6_2.IMAGE_SIZE, Chapter6_2.IMAGE_SIZE, Chapter6_2.NUM_CHANNELS], name='x-input')

        # 这个不对了，因为labels是一个数字
        # y_ = tf.placeholder(tf.float32, [None, Chapter5_20.OUTPUT_NODE], name='y-input')

        y_ = tf.placeholder(tf.int64, [mnist.validation.num_examples, ], name='y-input')

        reshaped_xs = np.reshape(xs, (
            mnist.validation.num_examples, Chapter6_2.IMAGE_SIZE, Chapter6_2.IMAGE_SIZE, Chapter6_2.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}
        print(mnist.validation.labels)
        y = Chapter6_2.inference(x,False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(Chapter6_3.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(Chapter6_3.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("after %s training step(s). validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("./Data/mnist")
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
