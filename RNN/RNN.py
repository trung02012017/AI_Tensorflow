import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)


def get_goodletrace_data(path, aspects):
    names = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId", "meanCPUUsage", "canonical memory usage",
             "AssignMem", "unmapped_cache_usage", "page_cache_usage", "max_mem_usage", "mean_diskIO_time",
             "mean_local_disk_space", "max_cpu_usage", "max_disk_io_time", "cpi", "mai", "sampling_portion", "agg_type",
             "sampled_cpu_usage"]
    df = pd.read_csv(path, names=names)
    data = df.loc[:, aspects].values

    a_max = np.amax(data, axis=0)
    a_min = np.amin(data, axis=0)

    normalized_data = (data - a_min)/(a_max - a_min)

    return normalized_data, a_max, a_min


def get_data_samples(data, n_slidings, predicted_aspect, rate):

    sliding = n_slidings
    n_samples = data.shape[0]
    n_aspects = data.shape[1]

    data_samples = n_samples - sliding
    data_feed = np.zeros((data_samples, sliding, n_aspects))

    for i in range(data_samples):
        a = i
        b = i + 3
        data_point = data[a:b, :]
        data_feed[i] += data_point

    n_test = int(data_samples/rate)
    n_train = int(data_samples - n_test)

    x_train = data_feed[0:n_train, :, :].reshape((n_train, sliding, n_aspects))
    x_test = data_feed[n_train:data_samples, :, :].reshape((n_test, sliding, n_aspects))

    y_feed = data[sliding:n_samples, :]
    if predicted_aspect == "meanCPUUsage":
        y_train = y_feed[0:n_train, 0].reshape((n_train, 1))
        y_test = y_feed[n_train:data_samples, 0].reshape((n_test, 1))

    if predicted_aspect == "canonical memory usage":
        y_train = y_feed[0:n_train, 1].reshape((n_train, 1))
        y_test = y_feed[n_train:data_samples, 1].reshape((n_test, 1))


    return x_train, y_train, x_test, y_test


def model_rnn(X, n_lstm_cells):
    size = n_lstm_cells
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=size)

    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,
        X,
        initial_state=None,
        dtype=tf.float32,
        time_major=False,
    )

    output = tf.layers.dense(outputs[:, -1, :], 1)

    return output


def main():

    path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
    aspects = ["meanCPUUsage", "canonical memory usage"]
    predicted_aspect = "meanCPUUsage"
    n_slidings = 3
    rate = 5

    nor_data, amax, amin = get_goodletrace_data(path, aspects)
    x_train, y_train, x_test, y_test = get_data_samples(nor_data, n_slidings, predicted_aspect, rate)

    batch_size = 32
    learning_rate = 0.005
    num_epochs = 100
    rnn_cellsize = 32
    num_batches = int(x_train.shape[0] / batch_size)

    timestep = n_slidings
    input_dim = len(aspects)
    X = tf.placeholder(tf.float32, [None, timestep, input_dim])
    y = tf.placeholder(tf.float32, [None, 1])

    output = model_rnn(X, rnn_cellsize)

    loss = tf.reduce_mean(tf.squared_difference(output, y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(num_epochs):
            for j in range(num_batches):
                a = batch_size * j
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
        output_test = output_test * (amax[0] - amin[0]) + amin[0]
        y_test_act = y_test*(amax[0] - amin[0]) + amin[0]

        loss_test_act = np.mean(np.abs(output_test - y_test_act))
        print(loss_test_act)

        plt.plot(y_test_act, 'r-', label="y actual")
        plt.plot(output_test, 'b-', label="y predict")
        plt.legend()
        plt.title("Single layer LSTM :  CPU & RAM => CPU ... Loss = %f" % (loss_test_act))
        plt.show()


if __name__ == '__main__':
    main()