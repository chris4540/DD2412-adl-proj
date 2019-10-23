# WRN code from https://github.com/EricAlcaide/keras-wrn/blob/master/keras_wrn/wrn.py

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Conv2DTranspose

def main_block(x, filters, n, strides, dropout, block_name, x0):
    # Normal part
    x_res = Conv2D(filters, (3, 3), strides=strides, padding="same",
                   use_bias=False)(x) # , kernel_regularizer=l2(5e-4)
    x_res = BatchNormalization()(x_res)
    x_res = Activation('relu')(x_res)
    x_res = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x_res)
    # Alternative branch
    x = Conv2D(filters, (1, 1), strides=strides, use_bias=False)(x)
    # Merge Branches
    if x0 is None or filters==32:
        x = Add()([x_res, x])
    else:
        x = Add()([x_res, x0])

    for i in range(n - 1):
        # Residual conection
        x_res = BatchNormalization()(x)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x_res)
        # Apply dropout if given
        if dropout: x_res = Dropout(dropout)(x)
        # Second part
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x_res)
        # Merge branches
        x = Add()([x, x_res])

    # Inter block part
    x = BatchNormalization()(x)
    x = Activation('relu', name=block_name)(x)
    return x


def build_model(input_dims, output_dim, n, k, act="relu", dropout=None):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
            - input_dims: input dimensions for the model
            - output_dim: output dimensions for the model
            - dropout: dropout rate - default=0 (not recomended >0.3)
            - act: activation function - default=relu. Build your custom
                   one with keras.backend (ex: swish, e-swish)
    """
    # Ensure n & k are correct
    assert (n - 4) % 6 == 0
    # assert k%2 == 0
    n = (n - 4) // 6
    # This returns a tensor input to the model
    inputs = Input(shape=(input_dims))

    # Head of the model
    x0 = Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x0)
    x = Activation('relu')(x)

    # 3 Blocks (normal-residual)
    x = main_block(x, 16 * k, n, (1, 1), dropout, 'act1', x0)  # 0
    x = main_block(x, 32 * k, n, (2, 2), dropout, 'act2', None)  # 1
    x = main_block(x, 64 * k, n, (2, 2), dropout, 'act3', None)  # 2

    # Final part of the model
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    outputs = Dense(output_dim, activation="softmax", name='final_output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
