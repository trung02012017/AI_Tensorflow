import numpy as np
import matplotlib.pyplot as plt

# Define input data
X_data = np.arange(100, step=0.1)
y_data = X_data + 20 * np.sin(X_data / 10)


import tensorflow as tf

n_samples = len(X_data)
batch_size = 100

X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weight", (1,1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,), initializer=tf.random_normal_initializer())

    y_pre = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pre)**2)/(2*n_samples)

optimizer = tf.train.AdamOptimizer().minimize(loss)
n_loops = 5000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step_idx in range(n_loops):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        _, loss_val = sess.run([optimizer, loss], feed_dict={X: X_batch, y: y_batch})

        if (step_idx+1)%1000 == 0:
            print(loss_val)
