import numpy as np
import tensorflow as tf

# 3 batchs 'hello', 'eolll', 'lleel'
x_data = np.array([[[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]],
                   [[11., 12.], [13., 14.], [15., 16.], [17., 18.], [19., 20.]],
                   [[21., 22.], [23., 24.], [25., 26.], [27., 28.], [29., 30.]]], dtype=np.float32)

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(outputs.eval())
