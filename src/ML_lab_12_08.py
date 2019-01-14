import numpy as np
import tensorflow as tf

seq_length = 7

def MinMaxScaler(xy_):
    min_xy = np.min(xy_, axis=0)
    max_xy = np.max(xy_, axis=0)
    new_xy = (xy_ - min_xy) / (max_xy - min_xy)
    return new_xy

# Open, High, Low, Close, Volume
xy = np.loadtxt('../data/data-02-stock_daily.csv', delimiter=',')
print('xy shape: {}'.format(xy.shape))

xy = xy[::-1]  # reverse order (chornically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]
print('xy shape: {}'.format(xy.shape))
print('y shape: {}'.format(y.shape))

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split into train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

class RNNModel(object):
    def __init__(self, seq_length_=7, learning_rate=0.01):
        self.timesteps = seq_length_
        self.data_dim = 5
        self.hidden_dim = 10
        self.output_dim = 1
        self.learning_rate = learning_rate

        self._network()

    def _network(self):
        # input placeholders
        self.X = tf.placeholder(tf.float32, [None, self.timesteps, self.data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)
        outputs, _state = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        self.Y_pred = tf.layers.dense(outputs[:, -1], self.output_dim, activation=None)
        # We use the last cell's output

        # cost/loss
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

class RegressionModel(object):
    def __init__(self, seq_length_=7, learning_rate=0.01):
        self.timesteps = seq_length_
        self.data_dim = 5
        self.hidden_dim = 10
        self.output_dim = 1
        self.learning_rate = learning_rate

        self._network()

    def _network(self):
        # input placeholders
        self.X = tf.placeholder(tf.float32, [None, self.timesteps, self.data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        x_flatten = tf.contrib.layers.flatten(self.X)
        outputs = tf.layers.dense(x_flatten, self.hidden_dim, activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.he_normal())
        self.Y_pred = tf.layers.dense(outputs, self.output_dim, activation=None,
                                  kernel_initializer=tf.initializers.he_normal())

        # cost/loss
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)


rnnModel = RNNModel(seq_length_=7)
regModel = RegressionModel(seq_length_=7)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, rnn_loss = sess.run([rnnModel.train_op, rnnModel.loss], feed_dict={rnnModel.X: trainX, rnnModel.Y: trainY})
    _, reg_loss = sess.run([regModel.train_op, regModel.loss], feed_dict={regModel.X: trainX, regModel.Y: trainY})
    print('iter: {}, rnn_loss: {:.3f}, reg_loss:{:.3f}'.format(i, rnn_loss, reg_loss))

rnnPredicts = sess.run(rnnModel.Y_pred, feed_dict={rnnModel.X: testX})
regPredicts = sess.run(regModel.Y_pred, feed_dict={regModel.X: testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(rnnPredicts)
plt.plot(regPredicts)
plt.legend(['Real', 'RNN Predict', 'Regression Predict'])

plt.show()
