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
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D


# def NavieGenerator(input_dim=100):
#     model = tf.keras.Sequential()
#     model.add(Dense(8 * 8 * 128, input_shape=(input_dim,)))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())

#     model.add(Reshape((8, 8, 128)))
#     assert model.output_shape == (None, 8, 8, 128)

#     model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
#     assert model.output_shape == (None, 16, 16, 128)
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())

#     model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
#     assert model.output_shape == (None, 32, 32, 64)
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())

#     model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     assert model.output_shape == (None, 32, 32, 3)
#     return model
def NavieGenerator(input_dim=100):
    model = tf.keras.Sequential()
    model.add(Dense(8 * 8 * 128, input_shape=(input_dim,)))
    model.add(Reshape((8, 8, 128), input_shape=(8*8*128,)))
    model.add(BatchNormalization(input_shape=(8, 8, 128)))
    assert model.output_shape == (None, 8, 8, 128)

    #
    model.add(UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 16, 16, 128)

    #
    model.add(UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 32, 32, 64)

    # Output
    model.add(Conv2D(3, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    assert model.output_shape == (None, 32, 32, 3)

    return model

if __name__ == "__main__":
    import numpy as np
    gen = NavieGenerator(input_dim=100)
    z = np.random.normal(size=(128, 100))
    img = gen(z)
    print(img.shape)
