import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def plot_compare(loss_list: list, ylim=None, title=None) -> None:
    bn_ = [i[0] for i in loss_list]
    nn_ = [i[1] for i in loss_list]

    plt.figure(figsize=(15, 10))
    plt.plot(bn_, label='With BN')
    plt.plot(nn_, label='Without BN')

    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    plt.legend()
    plt.grid('on')
    plt.show()


class Model:
    """ Network Model Class

    Note that this class has only the constructor.
    The actual model is defined inside the constructor.

    Attributes
    -------------
    X: tf.float32
        This is a tensorflow placeholder for MNIST images
        Expected shape is [None, 784]

    y: tf.float32
        This is a tensorflow placeholder for MNIST labels (one hot encoded)
        Expected shape is [None, 10]

    mode: tf.bool
        This is used for the batch normalization
        It's 'True' at training time and 'False' at test time

    loss: tf.float32
        The loss function is a softmax cross entropy

    trian_op:
        This is simply the training op that minimizes the loss

    accuracy: tf.float32
        The accuracy operation

    Examples
    --------------
    # >>> model = Model("Batch Norm", 32, 10)

    """
    def __init__(self, name, input_dim_, output_dim_, conv_dims=[32, 64, 128], hidden_dims=[256, 10],
                  use_batchnorm=True, drop_ratio=0.7, activation_fn=tf.nn.relu, optimizer=tf.train.AdamOptimizer,
                  lr=0.001):
        """ Constructor
        parameters
        -------------
        name: str
            The name of this network
            The entire network will be created under 'tf.variable_scope(name)'

        input_dim_: int
            The input dimension
            In this example, 784

        output_dim_: int
            The number of output labels
            There are 10 labels

        hidden_dims: list (default: [32, 32])
            len(hidden_dims) = number of layers
            each element is the number of hidden units

        use_batchnorm: bool (default: True)
            If true, it will crate teh batchnormalization layer

        activation_fn: TF functions (default: tf.nn.relu)
            Activation Function

        optimizer: TF optimizer (default: tf.train.AdamOptimizer)
            Optimizer Function

        lr: float (default: 0.01)
            Learning rate
        """
        with tf.variable_scope(name):
            # placeholders are defiend
            self.X = tf.placeholder(tf.float32, [None, input_dim_], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_dim_], name='y')
            self.mode = tf.placeholder(tf.bool, name='train_mode')

            # Loop over convoluation layers
            net = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            for i, conv_dim in enumerate(conv_dims):
                with tf.variable_scope('conv_{}'.format(i)):
                    net = tf.layers.conv2d(inputs=net, filters=conv_dim, kernel_size=[3, 3], padding='same',
                                           kernel_initializer=tf.initializers.he_normal())
                    if use_batchnorm:
                        net = tf.layers.batch_normalization(net, training=self.mode)
                    net = activation_fn(net)
                    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2, 2])
                    net = tf.layers.dropout(inputs=net, rate=drop_ratio, training=self.mode)

            net = tf.contrib.layers.flatten(net)

            # Loop over fully connection layers
            for i, hidden_dim in enumerate(hidden_dims):
                with tf.variable_scope('fc_{}'.format(i + len(conv_dims))):
                    net = tf.layers.dense(inputs=net, units=hidden_dim, kernel_initializer=tf.initializers.he_normal())
                    if use_batchnorm:
                        net = tf.layers.batch_normalization(net, training=self.mode)
                    net = activation_fn(net)
                    net = tf.layers.dropout(inputs=net, rate=drop_ratio, training=self.mode)

            hypothesis = tf.identity(net, name='logits')
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.y)
            self.loss = tf.reduce_mean(self.loss, name='loss')

            # When using the batchnormalization layers
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer(lr).minimize(self.loss)

            # Accuracy etc
            softmax = tf.nn.softmax(net, name='softmax')
            self.accuracy = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, dtype=tf.float32)) * 100.


class Solver:
    """ Solver class

    This class will contain the model class and session

    Attributes
    --------------
    model: Model class
    sess: TF session

    Methods:
    -------------
    train(X, y)
        Run the train_op and Returns the loss

    evaluate(X, y, batch_size=None)
        Returns "Loss" and "Accuracy"
        If batch_size is given, it's computed using batch_size
        because GPU memories cannot handle the entire training data at once

    Example
    -------------
    # >>> sess = tf.InteractiveSession()
    # model = Model('batchNorm', 32, 10)
    # solve = Solver(sess, model)

    # Train
    # >>> solver.trian(X, y)

    # Evaluate
    # >>> solver.evaluate(X, y)

    """
    def __init__(self, sess_, model):
        self.model = model
        self.sess = sess_

    def train(self, x, y):
        feed = {
            self.model.X: x,
            self.model.y: y,
            self.model.mode: True
        }

        train_op = self.model.train_op
        loss_ = self.model.loss

        return self.sess.run([train_op, loss_], feed_dict=feed)

    def evaluate(self, X, y, batch_size_=None):
        if batch_size_:
            num_data = X.shape[0]

            total_loss = 0
            total_acc = 0

            for i in range(0, num_data, batch_size_):
                X_batch_ = X[i:i+batch_size_]
                y_batch_ = y[i:i+batch_size_]

                feed = {
                    self.model.X: X_batch_,
                    self.model.y: y_batch_,
                    self.model.mode: False
                }

                loss_ = self.model.loss
                accuracy = self.model.accuracy

                step_loss, step_acc = self.sess.run([loss_, accuracy], feed_dict=feed)

                total_loss += step_loss * X_batch_.shape[0]
                total_acc += step_acc * X_batch_.shape[0]

            total_loss /= num_data
            total_acc /= num_data

            return total_loss, total_acc

        else:
            feed = {
                self.model.X: X,
                self.model.y: y,
                self.model.mode: False
            }

            loss_ = self.model.loss
            accuracy = self.model.accuracy

            return self.sess.run([loss_, accuracy], feed_dict=feed)


mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
print('Training data shape: {}'.format(mnist.train.images.shape))

input_dim = 784
output_dim = 10
N = 55000

sess = tf.Session()

# We create two models: one with the batch norm and other without
bn = Model('batchnorm', input_dim, output_dim, use_batchnorm=True)
nn = Model('no_norm', input_dim, output_dim, use_batchnorm=False)

# We create two solvers: to train both models at the same time for comparison
# Usually we only need on sovler class
bn_solver = Solver(sess, bn)
nn_solver = Solver(sess, nn)

epoch_n = 10
batch_size = 32

# Save losses and Accuracies every epoch
train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(epoch_n):
    for idx in range(int(N/batch_size)):
        X_batch, y_batch = mnist.train.next_batch(batch_size)

        _, bn_loss = bn_solver.train(X_batch, y_batch)
        _, nn_loss = nn_solver.train(X_batch, y_batch)

    print('Epoch: {} finished'.format(epoch))

    b_loss, b_acc = bn_solver.evaluate(mnist.train.images, mnist.train.labels, batch_size)
    n_loss, n_acc = nn_solver.evaluate(mnist.train.images, mnist.train.labels, batch_size)

    # Save train losses/accs
    train_losses.append([b_loss, n_loss])
    train_accs.append([b_acc, n_acc])
    print('[Epoch {}-TRAIN] Batchnorm Loss(Acc): {:.5f}({:.2f}%) vs '
          'No Batchnorm Loss(Acc): {:.5f}({:.2f}%)'.format(epoch, b_loss, b_acc, n_loss, n_acc))

    b_lss, b_acc = bn_solver.evaluate(mnist.validation.images, mnist.validation.labels)
    n_loss, n_acc = nn_solver.evaluate(mnist.validation.images, mnist.validation.labels)

    # save valid losses/accs
    valid_losses.append([b_loss, n_loss])
    valid_accs.append([b_acc, n_acc])
    print('[Epoch {}-VALID] Batchnorm Loss(Acc): {:.5f}({:.2f}%) vs '
          'No Batchnorm Loss(Acc): {:.5f}({:.2f}%)'.format(epoch, b_loss, b_acc, n_loss, n_acc))

loss, acc = bn_solver.evaluate(mnist.test.images, mnist.test.labels)
print('With Batchnorm Loss: {:.5f}, Acc: {:.2f}%'.format(loss, acc))
loss, acc = nn_solver.evaluate(mnist.test.images, mnist.test.labels)
print('Without Batchnorm Loss: {:.5f}, ACC: {:.2f}%'.format(loss, acc))

plot_compare(train_losses, title='Training Loss ast Epoch')
plot_compare(train_accs, ylim=[0., 100.], title='Training Acc at Epoch')
plot_compare(valid_losses, title='Validation Loss at Epoch')
plot_compare(valid_accs, ylim=[0., 100.], title='Validation Acc at Epoch')
