import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)


class Model:
    def __init__(self, sess_, name):
        self.sess = sess_
        self.name = name
        self._build_net()

    def _build_net(self, learning_rate=0.001):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.get_variable('W1', shape=[3, 3, 1, 32], initializer=tf.initializers.he_normal())
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            W2 = tf.get_variable('W2', shape=[3, 3, 32, 64], initializer=tf.initializers.he_normal())
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            L2 = tf.contrib.layers.flatten(L2)

            W3 = tf.get_variable('W3', shape=[L2.get_shape().as_list()[1], 10], initializer=tf.initializers.he_normal())
            b3 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L2, W3) + b3

            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

            accuracy_bool = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(accuracy_bool, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})

training_epochs = 10
batch_size = 100
models = []
num_models = 7

# initialize
sess = tf.Session()
for m in range(num_models):
    models.append(Model(sess, 'model' + str(m)))
sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch: {}, cost = {}'.format(str(epoch + 1).zfill(3), avg_cost_list))

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy: {:.2f}%'.format(m.get_accuracy(mnist.test.images, mnist.test.labels) * 100))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

print('Ensemble accuracy: {:.2f}%'.format(sess.run(ensemble_accuracy) * 100))
