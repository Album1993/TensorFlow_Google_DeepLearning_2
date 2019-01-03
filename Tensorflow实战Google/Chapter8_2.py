import numpy as np
import tensorflow as tf

import matplotlib as mpl
# 没有GUI时使用matplotlib绘图
# mpl.use('Agg')

from matplotlib import pyplot as plt

HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数。
NUM_LAYERS = 2  # LSTM的层数。
TIMESTEPS = 10  # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000  # 训练轮数。
BATCH_SIZE = 32  # batch大小。
TRAINING_EXAMPLES = 10000  # 训练数据个数。
TESTING_EXAMPLES = 1000  # 测试数据个数。
SAMPLE_GAP = 0.01  # 采样间隔。


def generate_data(seq):
    X = []
    y = []

    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    # 不是根据X轴的坐标定的
    for i in range(len(seq) - TIMESTEPS):
        # print("x:", [seq[i:i + TIMESTEPS]])
        # print("y:",[seq[i + TIMESTEPS]])
        X.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

print("test start", test_start)
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
print("test end", test_end)
train_X, train_y = generate_data(np.cos(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(
    np.cos(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
print(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32))
print(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
print(len(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32))))


# print(train_X)
# print(train_y)


def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    output = outputs[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op


def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    predictions = []
    labels = []

    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze() + 1

    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))

    print("roor mean Square error is : %f" % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label="real_sin")

    plt.legend()
    plt.show()




ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)

X, y = ds.make_one_shot_iterator().get_next()

with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(X, y, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Evaluate model before training")

    run_eval(sess, test_X, test_y)

    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train step: " + str(i) + ",loss" + str(l))

    print("Evaluate model after training")

    run_eval(sess, test_X, test_y)

