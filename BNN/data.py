import pandas as pd
import numpy as np


def get_goodletrace_data(path, aspects):
    names = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId", "meanCPUUsage", "canonical memory usage",
             "AssignMem", "unmapped_cache_usage", "page_cache_usage", "max_mem_usage", "mean_diskIO_time",
             "mean_local_disk_space", "max_cpu_usage", "max_disk_io_time", "cpi", "mai", "sampling_portion", "agg_type",
             "sampled_cpu_usage"]
    df = pd.read_csv(path, names=names)
    data = df.loc[:, aspects].values

    a_max = np.amax(data, axis=0)
    a_min = np.amin(data, axis=0)

    # print(a_max, a_min)

    normalized_data = (data - a_min)/(a_max - a_min)
    # print(normalized_data[0:100])

    return normalized_data, a_max, a_min


def get_data_samples(data, n_slidings, predicted_aspect, rate):

    sliding = n_slidings
    n_samples = data.shape[0]
    n_aspects = data.shape[1]

    data_samples = n_samples - sliding
    data_feed = np.zeros((data_samples, sliding, n_aspects))

    for i in range(data_samples):
        a = i
        b = i + sliding
        data_point = data[a:b, :]
        data_feed[i] += data_point

    n_test = int(data_samples/(rate+1))
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

    # print(x_train[0:10])
    # print(y_train[0:10])
    return x_train, y_train, x_test, y_test


def get_data_decoder(x_train_encoder, x_test_encoder, n_slidings_decoder):
    n_slidings_encoder = x_train_encoder.shape[1]
    a = n_slidings_encoder - n_slidings_decoder
    x_train_decoder = x_train_encoder[:, a:n_slidings_encoder, :]
    x_test_decoder = x_test_encoder[:, a:n_slidings_encoder, :]
    return x_train_decoder, x_test_decoder
