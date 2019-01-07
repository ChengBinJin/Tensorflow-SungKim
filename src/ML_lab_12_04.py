import numpy as np
import tensorflow as tf

sentence = ("if you want to build a ship, don't drum up people togother to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

dataX = []
dataY = []

seq_length = 10

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

data_dim = len(char_set)
rnn_hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])   # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])   # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        l, _ = sess.run([loss, train_op], feed_dict={X: dataX, Y: dataY})
        results = sess.run(prediction, feed_dict={X: dataX})
        # print char using dic

        for idx, result in enumerate(results):
            print('result shape: {}'.format(result.shape))
            result_str = [char_set[c] for c in np.squeeze(result)]
            print(str(i).zfill(4), str(idx).zfill(3), "loss: ", l, " Prediction: ", ''.join(result_str))
