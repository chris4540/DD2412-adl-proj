import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Conv2DTranspose, Reshape

def generator(input_dimension=100):
    model = tf.keras.Sequential()
    model.add(Dense(8 * 8 * 128, input_shape=(input_dimension,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((8, 8, 128)))
    assert model.output_shape == (None, 8, 8, 128)

    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    assert model.output_shape == (None, 32, 32, 3)
    return model
