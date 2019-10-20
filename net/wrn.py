"""
Keras implementation of Wide Res Net

As we aim on develop from scratch, we should make something different

Reference:
https://github.com/EricAlcaide/keras-wrn/blob/master/keras_wrn/wrn.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras import Model
from tensorflow.keras import layers

class BasicBlocks(layers.Layer):
    """
    basic - with two consecutive 3 × 3 convolutions with batch normalization and ReLU
    preceding convolution: conv3×3 and conv3×3

    Basic block structure:
        - Batch Normalization
        - ReLU activation
        - Conv 2d
        - Batch Normalization
        - ReLU activation
        - Conv 2d

    """
    def __init__(self, in_planes, out_planes, strides, dropRate=0.0):
        super(BasicBlocks, self).__init__()

        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        # bias is off as we have batch norm already, it always zero
        self.conv2d1 = Conv2D(out_planes, kernel_size=3, strides=strides, padding="same", use_bias=False)

        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')
        # bias is off as we have batch norm already, it always zero
        # strides = 1 s.t. keep the size unchange
        self.conv2d2 = Conv2D(out_planes, kernel_size=3, strides=1, padding="same", use_bias=False)

    def call(self, input_):
        # First conv 3x3
        out = self.bn1(input_)
        out = self.relu1(out)
        out = self.conv2d1(out)

        # Second conv 3x3
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2d2(out)

        return out

class WideResNet(Model):
    def __init__(self, output_dim=10):
        super(WideResNet, self).__init__()
        widen_factor = 1
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        # 1st conv before any network block
        self.conv1 = Conv2D(nChannels[0], kernel_size=3, strides=1, padding="same")

        # TODO: fix this
        self.block1 = BasicBlocks(nChannels[0], nChannels[1], 1)
        self.block2 = BasicBlocks(nChannels[1], nChannels[2], 2)
        self.block3 = BasicBlocks(nChannels[2], nChannels[3], 2)

        self.avg_pool = AveragePooling2D((8, 8))
        self.flatten = Flatten()
        self.fc = Dense(output_dim)
        self.softmax = Softmax()

    def call(self, input_):
        out = self.conv1(input_)
        #
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        #
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

if __name__ == "__main__":
    model = WideResNet()

    input_dims = (1, 32, 32, 3)  # (batch, x, y, nchannels)

    model.build(input_shape=input_dims)
    # test_input = tf.random.normal(input_dims)
    # out = model(test_input)
    # print(out)
    # model.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(model, "WRN-18-1.pdf", show_shapes=True)