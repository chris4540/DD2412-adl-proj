# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.

Reference:
    - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
    - https://towardsdatascience.com/review-wrns-wide-residual-networks-image-classification-d3feb3fb2004
    - https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua

Notes:
    1. Used Pre-Activation ResNet
        performing batch norm and ReLU before convolution
        i.e. BN-ReLU-Conv
"""
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax
import tensorflow.keras.backend as K

def WideResidualNetwork(depth=28, width=8, dropout_rate=0.0,
                        input_shape=None,
                        classes=10, activation='softmax'):
    """Instantiate the Wide Residual Network architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.

        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.

        # Arguments
            depth: number or layers in the DenseNet
            width: multiplier to the ResNet width (number of filters)
            dropout_rate: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                "cifar10" (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `tf` dim ordering)
                or `(3, 32, 32)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.
        """

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')


    img_input = Input(shape=input_shape)

    x = __create_wide_residual_network(classes, img_input,
            depth=depth,
            width=width,
            dropout=dropout_rate,
            activation=activation)
    #
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='wide-resnet')

    return model

def __conv1_block(input_):

    x = Conv2D(16, (3, 3), padding='same')(input_)
    return x


def __basic_residual_basic_block(input_, nInputPlane, nOutputPlane, strides):
    """
    See [Wide Residual Networks] Figure 1(a); B(3, 3) implementation
    """

    # ==================
    # residual layers
    # ==================
    # Pre-Activation
    x = BatchNormalization()(input_)
    x = Activation('relu')(x)
    x = Conv2D(nOutputPlane, (3, 3), strides=strides, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nOutputPlane, (3, 3), strides=1, padding="same")(x)

    # ==================
    # shortcut
    # ==================
    if nInputPlane != nOutputPlane:
        # need a down sample the last layer output in shortcut
        init = BatchNormalization()(input_)
        init = Activation('relu')(init)
        init = Conv2D(nOutputPlane, (1, 1), strides=strides)(init)
    else:
        init = input_

    m = Add()([init, x])
    return m

def __residual_block_group(input_, nInputPlane, nOutputPlane, count, strides):
    """
    For stacking blocks
    """
    x = input_
    for i in range(count):
        if i == 0:
            # strides = nOutputPlane // nInputPlane
            x = __basic_residual_basic_block(x, nInputPlane, nOutputPlane, strides)
        else:
            x = __basic_residual_basic_block(x, nOutputPlane, nOutputPlane, strides=1)
    return x


def __create_wide_residual_network(nb_classes, img_input, depth=28,
                                   width=8, dropout=0.0, activation='softmax'):
    ''' Creates a Wide Residual Network with specified parameters

    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        width: Width of the network.
        dropout: Adds dropout if value is greater than 0.0

    Returns:
        a Keras Model

    Notes:
        N is a number of blocks in group.
        minus 4 as we have
            1. 1 conv3x3 in conv1_block group
            2. 1 conv in each group for upsample / downsample in shortcut
               Each group has exactly one conv1x1 as shortcut size tunning
    '''

    N = (depth - 4) // 6

    x = __conv1_block(img_input)

    nChannels = [16, 16*width, 32*width, 64*width]

    # Block Group 2
    x = __residual_block_group(x, nChannels[0], nChannels[1], count=N, strides=1)


    # Block Group 3
    x = __residual_block_group(x, nChannels[1], nChannels[2], count=N, strides=2)

    # Block Group 3
    x = __residual_block_group(x, nChannels[2], nChannels[3], count=N, strides=2)


    # Avg pooling + classification
    x = GlobalAveragePooling2D()(x)
    # TODO
    # x = Dense(nb_classes, activation=activation)(x)
    x = Dense(nb_classes)(x)
    x = Softmax(axis=-1)(x)

    return x

if __name__ == "__main__":
    n = 10
    k = 2
    model = WideResidualNetwork(n, k, input_shape=(32, 32, 3))
    model.summary()
    from tensorflow.keras.utils import plot_model
    plt_name = "new-WRN-{}-{}.pdf".format(n, k)
    plot_model(model, plt_name, show_shapes=True, show_layer_names=True)
