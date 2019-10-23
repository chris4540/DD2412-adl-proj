"""
The generator use to provide a training sample that teacher and students has
the most difference estimation distribution.

TODO:
    Understand how this structure comes

Quote:
    We use a generic generator with only three convolutional layers,
    and our input noise z has 100 dimensions
"""
import tensorflow as tf
import tensorflow.keras.backend as K
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Reshape


def NavieGenerator(input_dim=100):
    model = tf.keras.Sequential()
    model.add(Dense(8 * 8 * 128, input_shape=(input_dim,)))
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
    #model.add(BatchNormalization())
    assert model.output_shape == (None, 32, 32, 3)
    return model

# function alias
generator = NavieGenerator
