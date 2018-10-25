import data as data
import numpy as np
import tensorflow as tf
import pandas as pd
import os.path
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)


def model_rnn(X, n_lstm_cells, activation):
    size = n_lstm_cells

    if activation == "tanh":
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=size)
    if activation == "sigmoid":
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=size, activation=tf.nn.sigmoid)
    outputs, state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=X,
        dtype=tf.float32,
    )

    output = tf.layers.dense(outputs[:, -1, :], 1)

    return output


def main():

    path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
    aspects = ["meanCPUUsage", "canonical memory usage"]
    predicted_aspect = "meanCPUUsage"
    n_slidings = [3, 4, 5]
    batch_sizes = [16, 32]
    learning_rate = 0.005
    num_epochs = 120
    rnn_cellsizes = [4, 8, 16]
    activations = ["tanh", "sigmoid"]
    rate = 5
    result_file_path = 'result.csv'

    combination = []
    for n_sliding in n_slidings:
        for batch_size in batch_sizes:
            for rnn_cellsize in rnn_cellsizes:
                for activation in activations:
                    combination_i = [n_sliding, batch_size, rnn_cellsize, activation]
                    combination.append(combination_i)
    print(combination)

    for combination_i in combination:
        tf.reset_default_graph()
        print(combination_i)

        n_sliding = combination_i[0]
        batch_size = combination_i[1]
        rnn_unit = combination_i[2]
        activation = combination_i[3]

        nor_data, amax, amin = data.get_goodletrace_data(path, aspects)
        x_train, y_train, x_test, y_test = data.get_data_samples(nor_data, n_sliding, predicted_aspect, rate)
        x_train, y_train, x_valid, y_valid = data.getValidationSet(x_train, y_train, 5)

        loss_train_value = []
        loss_valid_value = []

        num_batches = int(x_train.shape[0]/batch_size)
        timestep = n_sliding
        input_dim = len(aspects)
        X = tf.placeholder(tf.float32, [None, timestep, input_dim])
        y = tf.placeholder(tf.float32, [None, 1])

        output = model_rnn(X, rnn_unit, activation)

        loss = tf.reduce_mean(tf.squared_difference(output, y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)

            pre_loss_valid = 100
            x = 0
            early_stopping_val = 5
            epoch_i = 0
            for i in range(num_epochs):
                for j in range(num_batches):
                    a = batch_size * j
                    b = a + batch_size
                    x_batch = x_train[a:b, :, :]
                    y_batch = y_train[a:b, :]
                    # print(x_batch.shape, y_batch.shape)

                    loss_j, _ = sess.run([loss, optimizer], feed_dict={X: x_batch,
                                                                       y: y_batch})
                loss_train_i = sess.run(loss, feed_dict={X: x_train,
                                                         y: y_train})
                loss_valid_i = sess.run(loss, feed_dict={X: x_valid,
                                                         y: y_valid})
                print(loss_train_i, loss_valid_i)
                loss_train_value.append(loss_train_i)
                loss_valid_value.append(loss_valid_i)

                if loss_valid_i > pre_loss_valid:
                    x = x+1
                    if x == early_stopping_val:
                        break
                else:
                    x = 0
                print(x)
                pre_loss_valid = loss_valid_i
                epoch_i+=1

            output_test = sess.run(output, feed_dict={X: x_test,
                                                      y: y_test})
            output_test = output_test * (amax[0] - amin[0]) + amin[0]
            y_test_act = y_test*(amax[0] - amin[0]) + amin[0]

            loss_test_act = np.mean(np.abs(output_test - y_test_act))
            print(loss_test_act)

            combination_x = [combination_i]
            result = {'combination': combination_x,
                      'loss': loss_test_act}

            df = pd.DataFrame(result)
            if not os.path.exists(result_file_path):
                columns = ['combination', 'loss']
                df[columns]
                df.to_csv('result.csv', index=False, columns=columns)
            else:
                with open('result.csv', 'a') as csv_file:
                    df.to_csv(csv_file, header=False, index=False)

            plt.figure(2)
            plt.plot(loss_valid_value, 'r-', label="loss validation")
            plt.plot(loss_train_value, 'b-', label="loss train")
            plt.legend()
            name = ''
            name += str(combination_i)
            name += ' epoch='
            name += str(epoch_i)
            name += ' loss='
            name += str(loss_test_act)
            name += '.png'
            print(name)
            plt.savefig(name)
            plt.clf()


if __name__ == '__main__':
    main()
