"""
For data preprocessing
"""
import tensorflow as tf
import numpy as np
# import tensorflow.keras as keras

def standardize_data(data):
    ret = data.astype('float32') / 255.0
    # mean =
    # ret = (ret - ret.mean(axis=0)) / (ret.std(axis=0))
    # return ret

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = standardize_data(x_train)
    x_test = standardize_data(x_test)

    # normalized with train mean and std
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)

    # normalize
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return (x_train, y_train), (x_test, y_test)

def balance_sampling(data, lables_, data_per_class=200):

    # eps value to increase a bit of acceptance prob
    eps = 1e-4

    # Checking the shape of input
    n_data = data.shape[0]
    assert n_data == lables_.shape[0]

    lables = lables_.reshape(-1)
    cls_labels = np.unique(lables)
    nclasses = len(cls_labels)

    if nclasses*data_per_class > n_data:
        raise ValueError("Unable to sample data, the data per class is too large")

    # build a quota of classes first
    qouta_table = {i: data_per_class for i in cls_labels}

    # acceptance prob.
    p = (1.0 + eps) / nclasses

    selected_sample = []
    # loop over training data
    for i, label in enumerate(lables):
        if np.random.choice([True, False], p=[p, 1-p]):
            # check if still have quota for this class
            if label in qouta_table and qouta_table[label] > 0:
                selected_sample.append(i)
                qouta_table[label] -= 1

                # remove the label from qouta table if goes to zero
                if qouta_table[label] == 0:
                    qouta_table.pop(label, None)

        if not qouta_table:
            break

    # contruct the results
    sample_data = data[selected_sample, :]
    sample_lables = lables_[selected_sample, :]

    return sample_data, sample_lables

def to_categorical(labels):
    """
    Convert class vector to one-hot matrix

    Wrapper of tf.keras.utils.to_categorical
    """
    return tf.keras.utils.to_categorical(labels)

# ------------------------------------------------
# to be removed
# def get_cifar_data():
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

#     x_train = x_train.astype('float32') / 255
#     x_test = x_test.astype('float32') / 255
#     x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
#     x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

#     y_train = utils.to_categorical(y_train)
#     y_test = utils.to_categorical(y_test)
#     return x_train, y_train, x_test, y_test
