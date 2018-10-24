import numpy as np
import data as data
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)


def model_encoder(X, n_lstm_cells):
    n_layers = len(n_lstm_cells)
    cells = []
    for i in range(n_layers):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=n_lstm_cells[i], state_is_tuple=True)
        cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=X,
        dtype=tf.float32,
    )

    return outputs, state


def model_decoder(X, n_lstm_cells, initial_state):
    n_layers = len(n_lstm_cells)
    cells = []
    for i in range(n_layers):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=n_lstm_cells[i], state_is_tuple=True)
        cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=X,
        initial_state=initial_state,
        dtype=tf.float32,
    )

    return outputs, state


def main():
    path = "../Data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv"
    aspects = ["meanCPUUsage", "canonical memory usage"]
    predicted_aspect = "meanCPUUsage"
    n_slidings_encoder = 10
    n_slidings_decoder = 5
    rate = 4

    nor_data, amax, amin = data.get_goodletrace_data(path, aspects)
    x_train_encoder, y_train, x_test_encoder, y_test = data.get_data_samples(nor_data, n_slidings_encoder, predicted_aspect,
                                                                             rate)
    x_train_decoder, x_test_decoder = data.get_data_decoder(x_train_encoder, x_test_encoder, n_slidings_decoder)
    # print(x_train_encoder.shape, x_test_encoder.shape, x_train_decoder.shape, x_test_decoder.shape)

    batch_size = 32
    learning_rate = 0.005
    num_epochs = 100
    cell_encoder = [16, 32]
    cell_decoder = [16]
    num_batches = int(x_train_encoder.shape[0] / batch_size)

    timestep_encoder = n_slidings_encoder
    timestep_decoder = n_slidings_decoder
    input_dim = len(aspects)
    X_encoder = tf.placeholder(tf.float32, [None, timestep_encoder, input_dim])
    X_decoder = tf.placeholder(tf.float32, [None, timestep_decoder, input_dim])
    y = tf.placeholder(tf.float32, [None, 1])

    output_encoder, state_encoder = model_encoder(X_encoder, cell_encoder)
    last_state_encoder = state_encoder[-1]

    loss = tf.reduce_mean(tf.squared_difference(output_encoder, y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        output_encoder, state_encoder = sess.run([output_encoder, state_encoder], feed_dict={X_encoder: x_train_encoder,
                                                                                             y: y_train})

        c =
        print(output_encoder.shape, len(a), len(b))


if __name__ == '__main__':
    main()