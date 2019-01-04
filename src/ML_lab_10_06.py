import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_compare(loss_list: list, ylim=None, title=None) -> None:
    bn_ = [i[0] for i in loss_list]

    plt.figure(figsize=(15, 10))
    plt.plot(bn_, label='With BN')

    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    plt.legend()
    plt.grid('on')
    plt.show()

class Cifar10(object):
    def __init__(self):
        self.image_size = (32, 32, 3)
        self.num_valids = 5000

        self.cifar10_path = '../data/cifar10'
        self._load_cifar10()

    def _load_cifar10(self):
        import cifar10

        cifar10.data_path = self.cifar10_path
        # The CIFAR-10 data-set is about 163 MB and will be downloaded automatically if it is not
        # located in the given path.
        cifar10.maybe_download_and_extract()

        self.train_data, self.train_class, self.train_one_hot_class = cifar10.load_training_data()
        self.test_data, self.test_class, self.test_one_hot_class = cifar10.load_test_data()
        # validation dataset
        self.valid_data = self.train_data[:self.num_valids]
        self.valid_class = self.train_class[:self.num_valids]
        self.valid_one_hot_class = self.train_one_hot_class[:self.num_valids]

        # exclude validation data from training data
        self.train_data = self.train_data[self.num_valids:]
        self.train_class = self.train_class[self.num_valids:]
        self.train_one_hot_class = self.train_one_hot_class[self.num_valids:]

        # zero centring
        self.train_data = self.train_data * 2. - 1.
        self.valid_data = self.valid_data * 2. - 1.
        self.test_data = self.test_data * 2. - 1.

        self.num_trains = self.train_data.shape[0]
        self.num_tests = self.test_data.shape[0]

        print('Num. of the training data: {}'.format(self.num_trains))
        print('Num. of the validation data: {}'.format(self.num_valids))
        print('Num. of the test data: {}'.format(self.num_tests))

    def train_next_batch(self, batch_size_):
        indexes = np.random.choice(self.num_trains, batch_size_, replace=False)
        batch_imgs = self.train_data[indexes]
        batch_labels = self.train_one_hot_class[indexes]

        return batch_imgs, batch_labels


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
    def __init__(self, name, input_dim_, output_dim_, conv_dims=[64, 128, 256], hidden_dims=[1024, 10],
                  use_batchnorm=True, drop_ratio=0.7, activation_fn=tf.nn.relu, optimizer=tf.train.AdamOptimizer,
                  lr=0.01):
        """ Constructor
        parameters
        -------------
        name: str
            The name of this network
            The entire network will be created under 'tf.variable_scope(name)'

        input_dim_: (int, int, int)
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
            self.X = tf.placeholder(tf.float32, [None, *input_dim_], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_dim_], name='y')
            self.mode = tf.placeholder(tf.bool, name='train_mode')

            # Loop over convoluation layers
            net = self.X
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


input_dim = (32, 32, 3)
output_dim = 10
N = 55000

sess = tf.Session()

dataset = Cifar10()
bn = Model('batchnorm', input_dim, output_dim, use_batchnorm=True)
bn_solver = Solver(sess, bn)

epoch_n = 20
batch_size = 100

# Save losses and Accuracies every epoch
train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(epoch_n):
    for idx in range(int(N/batch_size)):
        X_batch, y_batch = dataset.train_next_batch(batch_size)

        _, bn_loss = bn_solver.train(X_batch, y_batch)

    print('\nEpoch: {} finished'.format(epoch))

    b_loss, b_acc = bn_solver.evaluate(dataset.train_data, dataset.train_one_hot_class, batch_size)

    # Save train losses/accs
    train_losses.append([b_loss])
    train_accs.append([b_acc])
    print('[Epoch {}-TRAIN] Batchnorm Loss(Acc): {:.5f}({:.2f}%)'.format(epoch, b_loss, b_acc))

    b_loss, b_acc = bn_solver.evaluate(dataset.valid_data, dataset.valid_one_hot_class, batch_size)

    # save valid losses/accs
    valid_losses.append([b_loss])
    valid_accs.append([b_acc])
    print('[Epoch {}-VALID] Batchnorm Loss(Acc): {:.5f}({:.2f}%)'.format(epoch, b_loss, b_acc))

loss, acc = bn_solver.evaluate(dataset.test_data, dataset.test_one_hot_class, batch_size)
print('With Batchnorm Loss: {:.5f}, Acc: {:.2f}%'.format(loss, acc))


plot_compare(train_losses, title='Training Loss ast Epoch')
plot_compare(train_accs, ylim=[0., 100.], title='Training Acc at Epoch')
plot_compare(valid_losses, title='Validation Loss at Epoch')
plot_compare(valid_accs, ylim=[0., 100.], title='Validation Acc at Epoch')
