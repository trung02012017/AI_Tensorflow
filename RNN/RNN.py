import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)


names = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId", "meanCPUUsage", "canonical memory usage",
         "AssignMem", "unmapped_cache_usage", "page_cache_usage", "max_mem_usage", "mean_diskIO_time",
         "mean_local_disk_space", "max_cpu_usage", "max_disk_io_time", "cpi", "mai", "sampling_portion", "agg_type",
         "sampled_cpu_usage"]

df = pd.read_csv("../Data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv", names=names)

data = df.loc[:, ["canonical memory usage"]].values
a_max = np.amax(data)
a_min = np.amin(data)

data_samples = data.shape[0]
test_size = int(data_samples/5 - 1)
a = data_samples - test_size
y_test_act = data[a:data_samples, :].reshape((test_size, 1))
print(y_test_act)

data = (data-a_min)/(a_max-a_min)

print(data.shape)

sliding = 3


def get_data(data, sliding):
    total_series = data.shape[0]
    n_samples = total_series - sliding

    x_train = np.zeros((n_samples, sliding))
    y_train = np.zeros((n_samples, 1))
    for i in range(n_samples):
        for j in range(sliding):
            x_train[i,j] += data[i+j, 0]

        y_train[i, 0] = data[i+sliding, 0]

    N = x_train.shape[0]
    test_size = int(N/5)
    a = N - test_size

    x_test = x_train[a:N, :]
    y_test = y_train[a:N, :]
    x_train = x_train[0:a, :]
    y_train = y_train[0:a, :]

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data(data, sliding)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_train.shape, y_train.shape)

batch_size = 64
learning_rate = 0.005
num_epochs = 100
rnn_cellsize = 32
num_batches = int(x_train.shape[0]/batch_size)

time_step = 3
input_size = 1

X = tf.placeholder(tf.float32, [None,time_step,1])
y = tf.placeholder(tf.float32, [None, 1])

rnncell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_cellsize)

outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnncell,
    X,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,
)

output = tf.layers.dense(outputs[:, -1, :], 1)

loss = tf.reduce_mean(tf.squared_difference(output, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(num_epochs):
        for j in range(num_batches):
            a = batch_size*j
            b = a + batch_size
            x_batch = x_train[a:b, :, :]
            y_batch = y_train[a:b, :]

            loss_j, _ = sess.run([loss, optimizer], feed_dict={X: x_batch,
                                                               y: y_batch})
        loss_i, _ = sess.run([loss, optimizer], feed_dict={X: x_train,
                                                           y: y_train})
        print(loss_i)

    output_test = sess.run(output, feed_dict={X: x_test,
                                              y: y_test})
    output_test = output_test*(a_max-a_min) + a_min

    loss_test_act = loss = np.mean(np.square(output_test - y_test_act))

    plt.plot(y_test_act, 'r', label="y actual")
    plt.plot(output_test, 'b', label="y predict")
    plt.show()

    print(loss_test_act)








