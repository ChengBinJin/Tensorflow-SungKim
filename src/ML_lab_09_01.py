import numpy as np
import tensorflow as tf

num_hiddens = 10

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W1 = tf.Variable(tf.random_normal([2, num_hiddens]), name='weight1')
b1 = tf.Variable(tf.random_normal([num_hiddens]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([num_hiddens, num_hiddens]), name='weight2')
b2 = tf.Variable(tf.random_normal([num_hiddens]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([num_hiddens, num_hiddens]), name='weight3')
b3 = tf.Variable(tf.random_normal([num_hiddens]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([num_hiddens, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W))
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))
            print('step: {}, cost: {}'.format(step, sess.run(cost, feed_dict={X: x_data, Y: y_data})))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
