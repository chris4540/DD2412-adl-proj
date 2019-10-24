"""
Example script to show the regularization loss

Run:
python code_trunk/test_losses.py
"""
import tensorflow as tf
# Must run this in order to have similar behaviour as TF2.0
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)

from net.wide_resnet import WideResidualNetwork
import numpy as np


if __name__ == "__main__":
    print("Running tensorflow version: ", tf.__version__)
    model = WideResidualNetwork(10, 1, input_shape=(32, 32, 3), weight_decay=0.0001)

    rand_input = np.random.rand(*(3, 32, 32, 3)).astype(np.float32)

    with tf.GradientTape() as tape:
        outputs = model(rand_input, training=True)
        print(outputs)
        reg_losses = model.losses
        print(type(reg_losses))
        reg_loss = tf.reduce_sum(reg_losses)
        print(reg_loss.numpy())
