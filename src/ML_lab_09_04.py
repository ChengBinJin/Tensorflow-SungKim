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
W2 = tf.get_variable('weight2', shape=[num_hiddens[0], num_hiddens[1]],
                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('weight3', shape=[num_hiddens[1], num_hiddens[2]],
                     initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('weight4', shape=[num_hiddens[2], nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable('bias1', shape=[num_hiddens[0]], initializer=tf.constant_initializer(0.))
b2 = tf.get_variable('bias2', shape=[num_hiddens[1]], initializer=tf.constant_initializer(0.))
b3 = tf.get_variable('bias3', shape=[num_hiddens[2]], initializer=tf.constant_initializer(0.))
b4 = tf.get_variable('bias4', shape=[nb_classes], initializer=tf.constant_initializer(0.))

with tf.name_scope('layer1'):
    layer1 = tf.matmul(X, W1) + b1
    W1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    layer1_hist = tf.summary.histogram('layer1', layer1)

with tf.name_scope('layer2'):
    layer2 = tf.matmul(layer1, W2) + b2
    W2_hist = tf.summary.histogram('weights2', W2)
    b2_hist = tf.summary.histogram('biases2', b2)
    layer2_hist = tf.summary.histogram('layer2', layer2)

with tf.name_scope('layer3'):
    layer3 = tf.matmul(layer2, W3) + b3
    W3_hist = tf.summary.histogram('weights3', W3)
    b3_hist = tf.summary.histogram('biases3', b3)
    layer3_hist = tf.summary.histogram('layer3', layer3)

with tf.name_scope('layer4'):
    # Hypothesis (using softmax)
    layer4 = tf.matmul(layer3, W4) + b4
    layer4_hist = tf.summary.histogram('layer4', layer4)

with tf.name_scope('cost'):
    hypothesis = tf.nn.softmax(layer4)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

with tf.name_scope('accuracy'):
    # Test model
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    accuracy_summ = tf.summary.scalar('accuracy', accuracy)

summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/mnist/fc_layers')

# parameters
training_epochs = 20
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorlFlow variables
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s, c, _ = sess.run([summary, cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(s, global_step=epoch*total_batch+i)
            avg_cost += c / total_batch

        print('Epoch: ', '%03d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test the model using test sets
    print("Accuracy: {:.2f}".format(accuracy.eval(
        session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}) * 100.))
