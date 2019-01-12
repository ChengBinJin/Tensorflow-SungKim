import numpy as np
import tensorflow as tf

def MinMaxScaler(xy_):
    min_xy = np.min(xy_, axis=0)
    max_xy = np.max(xy_, axis=0)
    new_xy = (xy_ - min_xy) / (max_xy - min_xy)
    return new_xy

timesteps = seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim =1
# Open, High, Low, Close, Volume
xy = np.loadtxt('../data/data-02-stock_daily.csv', delimiter=',')
print('xy shape: {}'.format(xy.shape))

xy = xy[::-1] # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
print('xy: {}'.format(xy))
x = xy
y = xy[:, [-1]]
print('y shape: {}'.format(y.shape))  # close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length] # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split into train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
Y_pred = tf.layers.dense(outputs[:, -1], output_dim, activation=None)
# We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    print(i, l)

testPredict = sess.run(Y_pred, feed_dict={X: testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)

plt.show()

