"""
Example for
    - joining blocks and
    - Residual connection on a convolution layer
Ref: https://keras.io/getting-started/functional-api-guide/
"""
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Softmax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Conv2D, Input


# # # input tensor for a 3-channel 256x256 image
# x = Input(shape=(256, 256, 3))
# # 3x3 conv with 3 output channels (same as input channels)
# y = Conv2D(3, (3, 3), padding='same')(x)
# # this returns x + y.
# z = tf.keras.layers.add([x, y])

# model = Model(x, z)
# print(model.summary())
# print(type(z))

model1 = Sequential()
model1.add(Input(768))
model1.add(Dense(500))
model1.add(Activation('relu'))
model1.add(Dropout(0.4))

# # # -----------------------------
model2 = Sequential()
model1.add(Input(500))
model2.add(Dense(300))
model2.add(Activation('relu'))
model2.add(Dropout(0.4))
model2.add(Dense(10))
model2.add(Activation('softmax'))

# print(model1.summary())
# print(model2.summary())

join_model = Model(model1.input, model2.output)