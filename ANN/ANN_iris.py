import pandas as pd
import numpy as np
import tensorflow as tf

data_raw = pd.read_csv("iris.csv").values
n_samples = data_raw.shape[0]
num_class = 3

x_raw = data_raw[:, 0:4]
y_raw = data_raw[:, 4]

x_train = x_raw
y_train = np.zeros((n_samples, num_class))

d = x_train.shape[1]

for i in range(n_samples):
    if y_raw[i] == 'Setosa':
        y_train[i, 0] = 1

    if y_raw[i] == 'Versicolor':
        y_train[i, 1] = 1

    if y_raw[i] == 'Virginica':
        y_train[i, 2] = 1

learning_rate = 0.001
epochs = 3000
batch_size = 30

n_hidden1 = 10
n_hidden2 = 15
n_inputs = d
n_classes = num_class

X = tf.placeholder(tf.float32, shape=([None, n_inputs]))
y = tf.placeholder(tf.float32, shape=([None, n_classes]))

weight = {
    'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x):
    layer1 = tf.add(tf.matmul(x, weight['h1']), biases['b1'])
    layer2 = tf.add(tf.matmul(layer1, weight['h2']), biases['b2'])
    out_layer = tf.add(tf.matmul(layer2, weight['out']), biases['out'])
    return out_layer

logits = multilayer_perceptron(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = x_train[indices], y_train[indices]

        _, loss_value = sess.run([optimizer, loss], feed_dict={X: X_batch, y: y_batch})

        if (i+1)%100 == 0:
            print(loss_value)




