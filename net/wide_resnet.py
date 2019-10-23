# -*- coding: utf-8 -*-
"""
Wide Residual Network models for Keras.

This implementation follows the implementation of the authors' lua version.

Missing info if only check with the author paper:
    1. Add batchnorm + relu before the avg_pool layer
    2. conv1x1 used to adjust the input output size.
       it is called down sampling in the paper
       no bn+relu before this down sample conv1x1

Reference:
    - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
    - https://towardsdatascience.com/review-wrns-wide-residual-networks-image-classification-d3feb3fb2004
    - https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
    - https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

Notes:
    1. Used Pre-Activation ResNet
        performing batch norm and ReLU before convolution
        i.e. BN-ReLU-Conv
TODO:
    2. model.add_lost

Visualize the network:
    ```
    $ python wide_resnet.py
    ```
"""
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class Identity(Layer):
    """
    Identity layer, like nn.Identity in pytorch
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def WideResidualNetwork(depth=28, width=8, dropout_rate=0.0,
                        input_shape=None, classes=10,
                        has_softmax=True, output_activations=False):
    """
    Builder function to make wide-residual network
    """

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')

    # make model name
    model_name = 'wide-resnet-{}-{}'.format(depth, width)


    img_input = Input(shape=input_shape)

    ret = __create_wide_residual_network(classes, img_input,
            depth=depth,
            width=width,
            dropout=dropout_rate,
            has_softmax=has_softmax,
            output_activations=output_activations,
            model_name=model_name)


    return ret

def __conv1_block(input_):
    """
    The first convolution layer of WRN.
    As the paper call it conv1 group, we call it conv1_block for convention
    """

    x = Conv2D(16, kernel_size=3, padding='same', use_bias=False)(input_)
    return x


def __basic_residual_basic_block(input_, nInputPlane, nOutputPlane, strides, dropout=0.0):
    """
    See [Wide Residual Networks] Figure 1(a); B(3, 3) implementation
    TODO:
        doc
    """

    # ==================
    # residual blocks
    # ==================
    # Pre-Activation
    x = BatchNormalization()(input_)
    x = Activation('relu')(x)
    x = Conv2D(nOutputPlane, kernel_size=3, strides=strides, padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Mentioned in the paper section 2.4
    # A dropout layer shoud be after ReLU to perturb batch normalization in the
    # next residual block and prevent it from overfitting.
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Conv2D(nOutputPlane, kernel_size=3, strides=1, padding="same", use_bias=False)(x)

    # ==================
    # short circuit
    # ==================
    if nInputPlane != nOutputPlane:
        init = Conv2D(nOutputPlane, kernel_size=1, strides=strides, use_bias=False)(input_)
    else:
        init = input_

    m = Add()([init, x])
    return m

def __residual_block_group(input_, nInputPlane, nOutputPlane, count, strides, dropout=0.0):
    """
    For stacking blocks
    TODO:
        doc
    """
    x = input_
    for i in range(count):
        if i == 0:
            x = __basic_residual_basic_block(
                    x, nInputPlane, nOutputPlane, strides, dropout=dropout)
        else:
            # As the first block in group resolved unequal input and output
            # on this block, the strides will be 1 for this
            x = __basic_residual_basic_block(
                x, nOutputPlane, nOutputPlane, strides=1, dropout=dropout)
    return x


def __create_wide_residual_network(nb_classes, img_input, depth=28,
                                   width=8, dropout=0.0, has_softmax=True,
                                   output_activations=False, model_name=None):
    ''' Creates a Wide Residual Network with specified parameters

    Args:
        nb_classes: Number of output classes
        img_input: Input layer
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

    # Block Group: conv2
    x = __residual_block_group(x, nChannels[0], nChannels[1],
                               count=N, strides=1, dropout=dropout)
    act1 = x
    # att1 = Identity(name='attention1')(x)  # Identity layer


    # Block Group: conv3
    x = __residual_block_group(x, nChannels[1], nChannels[2],
                               count=N, strides=2, dropout=dropout)
    act2 = x
    # att2 = Identity(name='attention2')(x)  # Identity layer

    # Block Group: conv4
    x = __residual_block_group(x, nChannels[2], nChannels[3],
                               count=N, strides=2, dropout=dropout)
    act3 = x
    # att3 = Identity(name='attention3')(x)  # Identity layer


    # Avg pooling + fully connected layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    # relu is a must add otherwise cannot train
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    # Final classification layer
    x = Dense(nb_classes, name='logits')(x)
    if has_softmax:
        x = Softmax(axis=-1)(x)

    # make model as the return
    if output_activations:
        ret = Model(inputs=img_input, outputs=[x, act1, act2, act3], name=model_name)
    else:
        ret = Model(inputs=img_input, outputs=x, name=model_name)

    return ret

# ========================================================

if __name__ == "__main__":
    pass
    # n = 40
    # k = 2
    # model = WideResidualNetwork(n, k, input_shape=(32, 32, 3), dropout_rate=0.0)
    # model.summary()
    # model.save_weights('test.h5')

    # model2 = WideResidualNetwork(n, k, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
    # model.load_weights('test.h5')
    # # from tensorflow.keras.utils import plot_model
    # # plt_name = "new-WRN-{}-{}.pdf".format(n, k)
    # # plot_model(model, plt_name, show_shapes=True, show_layer_names=True)
