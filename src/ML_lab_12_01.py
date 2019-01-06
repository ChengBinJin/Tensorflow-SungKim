import numpy as np
import tensorflow as tf

# One hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# one cell RNN input_dim (4) -> output_dim (2). sequnce: 5
hidden_size = 2
# cell = tf.contrib.rnn.BasicRNNCell(num_unis=hidden_size)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# x_data = np.array([[[1, 0, 0, 0]]], dtype=np.float32)
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)
print(x_data.shape)
print(x_data)

outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('outputs shape: {}'.format(outputs.get_shape().as_list()))
print(sess.run(outputs))

# print('states shape: {}'.format(states.get_shape().as_list()))
print(sess.run(states))