"""
For data preprocessing

"""
import tensorflow as tf
import numpy as np

class Cifar10NormalFactors:
    mean = None
    std = None

def get_cifar10_mean_std():
    if Cifar10NormalFactors.mean is None or Cifar10NormalFactors.std is None:
        get_cifar10_data()

    assert Cifar10NormalFactors.mean is not None
    assert Cifar10NormalFactors.std is not None
    return Cifar10NormalFactors.mean, Cifar10NormalFactors.std

def standardize_data(data):
    ret = data.astype('float32') / 255.0
    return ret

def get_cifar10_data():
    """
    Get cifar 10 data. Do mean and bias removal

    Return:
        mean and bias removed data
    """
    (x_train, y_train_labels), (x_test, y_test_labels) = tf.keras.datasets.cifar10.load_data()

    x_train = standardize_data(x_train)
    x_test = standardize_data(x_test)

    # normalized with train mean and std
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)

    # save down for later use
    Cifar10NormalFactors.mean = x_train_mean
    Cifar10NormalFactors.std = x_train_std

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    return (x_train, y_train_labels), (x_test, y_test_labels)

def get_fashion_mnist_data():
    """
    Get fashion_mnist data. Do mean and bias removal

    Return:
        mean and bias removed data
    """
    (x_train, y_train_labels), (x_test, y_test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'symmetric')
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'symmetric')
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    x_train = standardize_data(x_train)
    x_test = standardize_data(x_test)

    # normalized with train mean and std
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    return (x_train, y_train_labels), (x_test, y_test_labels)
# ==========================================================================
def balance_sampling(data, lables_, data_per_class=200):

    # eps value to increase a bit of acceptance prob
    eps = 1e-1

    # Checking the shape of input
    n_data = data.shape[0]
    assert n_data == lables_.shape[0]

    lables = lables_.reshape(-1)
    cls_labels = np.unique(lables)
    nclasses = len(cls_labels)

    if nclasses*data_per_class >= n_data:
        raise ValueError("Unable to sample data, the data per class is too large")

    # build a quota of classes first
    qouta_table = {i: data_per_class for i in cls_labels}

    # acceptance prob.
    p = 1.0 / nclasses + eps

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
    if i ==  n_data - 1:
        print('Warning: Sampled all data')

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

if __name__ == "__main__":
    (x_train, y_train_lbl), (x_test, y_test_lbl) = get_cifar10_data()
    x_train, y_train_lbl = balance_sampling(x_train, y_train_lbl, data_per_class=200)
    y_train = to_categorical(y_train_lbl)
    y_test = to_categorical(y_test_lbl)
