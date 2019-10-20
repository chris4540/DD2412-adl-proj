"""
Example to write pytorch style network
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

class SimpleNet(Model):
    """
    Simple network for testing keras/ tensorflow
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

if __name__ == "__main__":
    # Config input dim and output dim
    input_dims = (100, 32, 32, 2)  # (batch, x, y, nchannels)
    inputs = Input(shape=(input_dims))
    # Create an instance of the model
    model = SimpleNet()
    model.build(input_shape=input_dims)

    model.summary()