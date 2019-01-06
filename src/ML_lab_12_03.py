import numpy as np
import tensorflow as tf

def print_list(results_str):
    results_ = ""
    for _, c in enumerate(results_str):
        results_ += c

    return results_

sample = ' if you want you'
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idx

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]  # Y label sample(1 ~ n) hello: ello

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
sequence_length = len(sample) - 1  # number of lstm unfolding (unit #)
num_classes = len(idx2char)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
print('X_one_hot shape: {}'.format(X_one_hot.get_shape().as_list()))

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
        l, _ = sess.run([loss, train_op], feed_dict={X: x_data, Y: y_data})
        results = sess.run(prediction, feed_dict={X: x_data})
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(results)]
        print(str(i).zfill(4), "loss: ", l, " Prediction: {}".format(print_list(result_str)),
              "Correct: {}".format(sample))
