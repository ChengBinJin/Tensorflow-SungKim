import numpy as np
import tensorflow as tf

idx2char = ['h', 'i', 'e', 'l', 'o']    # h=0, i=1, e=2, l=3, o=4
x_data = [[0, 1, 0, 2, 3, 3]]           # hihell
x_one_hot = [[[1, 0, 0, 0, 0],          # h 0
              [0, 1, 0, 0, 0],          # i 1
              [1, 0, 0, 0, 0],          # h 0
              [0, 0, 1, 0, 0],          # e 2
              [0, 0, 0, 1, 0],          # l 3
              [0, 0, 0, 1, 0]]]         # l 3
y_data = [[1, 0, 2, 3, 3, 4]]           # ihello

hidden_size = 5         # output from the LSTM
input_dim = 5           # one-hot size
batch_size = 1          # one sentence
sequence_length = 6     # |ihello| = 6

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])               # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        l, _ = sess.run([loss, train_op], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(str(i).zfill(4), "loss: ", l, "prediction: ", result, "true Y: ", y_data)


        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPredicton str: {}".format(result_str))


