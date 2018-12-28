import numpy as np
import tensorflow as tf


t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t.ndim)  # rank
print(t.shape)  # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t.ndim)  # rank
print(t.shape)  # shape

with tf.Session() as sess:
    t = tf.constant([1, 2, 3, 4])
    print(tf.shape(t).eval())

    t = tf.constant([[1, 2],
                     [3, 4]])
    print(tf.shape(t).eval())

    t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                       [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
    print(tf.shape(t).eval())

    matrix1 = tf.constant([[1., 2.], [3., 4.]])
    matrix2 = tf.constant([[1.], [2.]])
    print("Metrix 1 shape: {}".format(matrix1.shape))
    print("Metrix 2 shape: {}".format(matrix2.shape))
    print(tf.matmul(matrix1, matrix2).eval())

    print((matrix1 * matrix2).eval())

    # Operations between the same shape
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2., 2.]])
    print((matrix1 + matrix2).eval())

    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant(3.)
    print((matrix1 + matrix2).eval())

    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant([[3., 4.]])
    print((matrix1 + matrix2).eval())

    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant([[3.], [4.]])
    print((matrix1 + matrix2).eval())

    print(tf.reduce_mean([1, 2], axis=0).eval())

    x = [[1., 2.],
         [3., 4.]]
    print(tf.reduce_mean(x).eval())

    print(tf.reduce_mean(x, axis=0).eval())
    print(tf.reduce_mean(x, axis=1).eval())
    print(tf.reduce_mean(x, axis=-1).eval())

    print(tf.reduce_sum(x).eval())
    print(tf.reduce_sum(x, axis=0).eval())
    print(tf.reduce_sum(x, axis=-1).eval())
    print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())

    x = [[0, 1, 2],
         [2, 1, 0]]
    print(tf.argmax(x, axis=0).eval())
    print(tf.argmax(x, axis=1).eval())
    print(tf.argmax(x, axis=-1).eval())

    t = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                  [[6, 7, 8],
                   [9, 10, 11]]])
    print(t.shape)
    print(tf.reshape(t, shape=[-1, 3]).eval())
    print(tf.reshape(t, shape=[-1, 1, 3]).eval())

    print(tf.squeeze([[0], [1], [2]]).eval())
    print(tf.expand_dims([0, 1, 2], axis=1).eval())

    print(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
    t = tf.one_hot([[0], [1], [2], [0]], depth=3)
    print(tf.reshape(t, shape=[-1, 3]).eval())

    print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
    print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

    x = [1, 4]
    y = [2, 5]
    z = [3, 6]
    # Pack along first dim.
    print(tf.stack([x, y, z]).eval())
    print(tf.stack([x, y, z], axis=1).eval())

    x = [[0, 1, 2],
         [2, 1, 0]]
    print(tf.ones_like(x).eval())
    print(tf.zeros_like(x).eval())

for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
