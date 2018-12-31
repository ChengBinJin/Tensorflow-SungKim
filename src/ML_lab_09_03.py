import numpy as np
import tensorflow as tf

num_hiddens = 10

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

with tf.name_scope('layer1'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    W1 = tf.Variable(tf.random_normal([2, num_hiddens]), name='weight1')
    b1 = tf.Variable(tf.random_normal([num_hiddens]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # 1: From TF graph, decide which tensors you want to log
    w1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    layer1_hist = tf.summary.histogram('layer1', layer1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([num_hiddens, num_hiddens]), name='weight2')
    b2 = tf.Variable(tf.random_normal([num_hiddens]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    # 1: From TF graph, decide which tensors you want to log
    W2_hist = tf.summary.histogram('weights2', W2)
    b2_hist = tf.summary.histogram('biases2', b2)
    layer2_hist = tf.summary.histogram('layer2', layer2)

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([num_hiddens, num_hiddens]), name='weight3')
    b3 = tf.Variable(tf.random_normal([num_hiddens]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    # 1: From TF graph, decide which tensors you want to log
    W3_hist = tf.summary.histogram('weights3', W3)
    b3_hist = tf.summary.histogram('biases3', b3)
    layer3_hist = tf.summary.histogram('layer3', layer3)

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_normal([num_hiddens, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='biases4')
    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W))
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    # 1: From TF graph, decide which tensors you want to log
    W4_hist = tf.summary.histogram('weights4', W4)
    b4_hist = tf.summary.histogram('biases4', b4)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('cost'):
    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_sum = tf.summary.scalar('cost', cost)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_sum = tf.summary.scalar('accuracy', accuracy)

#2: Merge all summaries
summary = tf.summary.merge_all()

#3: Create writer and add graph
writer = tf.summary.FileWriter('./logs/xor/xor_logs')

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    #4: Run summary merge and add_summary
    writer.add_graph(sess.graph)

    for step in range(10001):
        s, _ =  sess.run([summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s, global_step=step)

        if step % 100 == 0:
            # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))
            print('step: {}, cost: {}'.format(step, sess.run(cost, feed_dict={X: x_data, Y: y_data})))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
