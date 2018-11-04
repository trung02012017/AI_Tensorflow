import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path

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
        b = i + n_slidings
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


def getValidationSet(x_train, y_train, n):
    n_train = x_train.shape[0]
    n_valid = int(n_train/n)
    n_train_new = n_train - n_valid

    x_val = x_train[n_train_new:n_train, :, :]
    y_val = y_train[n_train_new:n_train, :].reshape((n_valid, 1))

    x_train_new = x_train[0:n_train_new, :, :]
    y_train_new = y_train[0:n_train_new, :].reshape((n_train_new, 1))

    return x_train_new, y_train_new, x_val, y_val


def saveData(combination_i, loss_test_act, loss_valid_value, loss_train_value, epoch_i, result_file_path,
             output_test, y_test_act, explained_variance_score, mean_absolute_error, mean_squared_error,
             median_absolute_error, r2_score, training_time):
    combination_x = [combination_i]
    result = {'combination': combination_x,
              'epoch': epoch_i,
              'training_time': training_time,
              'explained_variance_score': explained_variance_score,
              'mean_absolute_error': mean_absolute_error,
              'mean_squared_error': mean_squared_error,
              'median_absolute_error': median_absolute_error,
              'r2_score': r2_score}

    df = pd.DataFrame(result)
    if not os.path.exists(result_file_path):
        columns = ['combination', 'epoch', 'training_time', 'explained_variance_score', 'mean_absolute_error',
                   'mean_squared_error', 'median_absolute_error', 'r2_score']
        df[columns]
        df.to_csv('result_multi.csv', index=False, columns=columns)
    else:
        with open('result_multi.csv', 'a') as csv_file:
            df.to_csv(csv_file, mode='a', header=False, index=False)

    plt.figure(2)
    plt.plot(loss_valid_value, 'r-', label="loss validation")
    plt.plot(loss_train_value, 'b-', label="loss train")
    plt.legend()
    name = 'loss '
    name += str(combination_i)
    name += ' epoch='
    name += str(epoch_i)
    name += ' loss='
    name += str(loss_test_act)
    name += '.png'
    plt.savefig(name)
    plt.clf()

    plt.figure(2)
    plt.plot(y_test_act, 'r-', label="actual y")
    plt.plot(output_test, 'b-', label="predicted y")
    plt.legend()
    name = 'result '
    name += str(combination_i)
    name += ' epoch='
    name += str(epoch_i)
    name += ' loss='
    name += str(loss_test_act)
    # print(name)
    name += '.png'
    plt.savefig(name)
    plt.clf()

    return name