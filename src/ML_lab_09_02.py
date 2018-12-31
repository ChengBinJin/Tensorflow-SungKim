import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more inofrmation about the mnist dataset
mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)

nb_classes = 10
input_dim = 784
num_hiddens = [512, 256, 128]

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, input_dim])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable('weight1', shape=[input_dim, num_hiddens[0]],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('bias1', shape=[num_hiddens[0]], initializer=tf.constant_initializer(0.))
layer1 = tf.matmul(X, W1) + b1

W2 = tf.get_variable('weight2', shape=[num_hiddens[0], num_hiddens[1]],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('bias2', shape=[num_hiddens[1]], initializer=tf.constant_initializer(0.))
layer2 = tf.matmul(layer1, W2) + b2

W3 = tf.get_variable('weight3', shape=[num_hiddens[1], num_hiddens[2]],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('bias3', shape=[num_hiddens[2]], initializer=tf.constant_initializer(0.))
layer3 = tf.matmul(layer2, W3) + b3

W4 = tf.get_variable('weight4', shape=[num_hiddens[2], nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable('bias4', shape=[nb_classes], initializer=tf.constant_initializer(0.))
# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(layer3, W4) + b4)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 20
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorlFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch: ', '%03d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test the model using test sets
    print("Accuracy: {:.2f}".format(accuracy.eval(
        session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}) * 100.))
