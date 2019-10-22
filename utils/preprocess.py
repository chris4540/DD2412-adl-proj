"""
For data preprocessing
"""
import tensorflow.keras as keras

def get_cifar_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
    x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test

 