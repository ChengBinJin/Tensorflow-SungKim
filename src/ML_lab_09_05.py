import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

nb_classes = 10
input_dim = 784
num_hiddens = [512, 512, 512, 512, 10]
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, input_dim])
# 0-9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])
# dropout (keep_prob) rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# W1 = tf.Variable(tf.random_normal([input_dim, num_hiddens[0]]))
# W1 = tf.get_variable('W1', shape=[input_dim, num_hiddens[0]], initializer=tf.contrib.layers.xavier_initializer())
W1 = tf.get_variable('W1', shape=[input_dim, num_hiddens[0]], initializer=tf.initializers.he_normal())
b1 = tf.Variable(tf.random_normal([num_hiddens[0]]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([num_hiddens[0], num_hiddens[1]]))
# W2 = tf.get_variable('W2', shape=[num_hiddens[0], num_hiddens[1]], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape=[num_hiddens[0], num_hiddens[1]], initializer=tf.initializers.he_normal())
b2 = tf.Variable(tf.random_normal([num_hiddens[1]]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# W3 = tf.Variable(tf.random_normal([num_hiddens[1], nb_classes]))
# W3 = tf.get_variable('W3', shape=[num_hiddens[1], nb_classes], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape=[num_hiddens[1], num_hiddens[2]], initializer=tf.initializers.he_normal())
b3 = tf.Variable(tf.random_normal([num_hiddens[2]]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', shape=[num_hiddens[2], num_hiddens[3]], initializer=tf.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([num_hiddens[3]]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', shape=[num_hiddens[3], num_hiddens[4]], initializer=tf.initializers.he_normal())
b5 = tf.Variable(tf.random_normal([num_hiddens[4]]))
hypothesis = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# trian my model
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Eoch: {}, cost = {:.9f}'.format(str(epoch + 1).zfill(4), avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: {:.2f} %'.format(
    sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.}) * 100))
