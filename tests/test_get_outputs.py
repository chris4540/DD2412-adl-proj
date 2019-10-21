"""
How to run:
    cd <project root>
    python -m unittest tests/test_get_outputs.py

Ref:
https://guillaumegenthial.github.io/testing.html
"""

import unittest
from net.wide_resnet import WideResidualNetwork
from net.wide_resnet import get_intm_outputs_of
import tensorflow as tf
import numpy as np

class Test(unittest.TestCase):
    input_shape = (32, 32, 3)
    depth = 10
    width = 1
    n_classes = 10

    @classmethod
    def setUpClass(cls):
        cls.model = WideResidualNetwork(
                    cls.depth, cls.width,
                    classes=cls.n_classes,
                    input_shape=cls.input_shape)

    def test_shapes(self):
        batch_size = 128
        # create input values
        test_input = np.random.random([batch_size, *self.input_shape])
        with tf.Session():
            tf_input = tf.constant(test_input, dtype=tf.float32)
            interm_values = get_intm_outputs_of(self.model, tf_input, mode="eval")

        self.assertEqual(interm_values['logits'].shape, (batch_size, self.n_classes))

        # attention1
        shape = (batch_size, 32, 32, 16*self.width)
        self.assertEqual(interm_values['attention1'].shape, shape)

        # attention2
        shape = (batch_size, 16, 16, 32*self.width)
        self.assertEqual(interm_values['attention2'].shape, shape)

        # attention3
        shape = (batch_size, 8, 8, 64*self.width)
        self.assertEqual(interm_values['attention3'].shape, shape)



if __name__ == "__main__":
    pass

    # # create model
    # model = WideResidualNetwork(depth, width, input_shape=input_shape)

    # # create input values
    # test_input = np.random.random([128, *input_shape])

    # with tf.Session() as sess:
    #     tf_input = tf.constant(test_input, dtype=tf.float32)
    #     interm_values = get_intm_outputs_of(model, tf_input, mode="eval")

    # print(interm_values['logits'].shape)
    # print(interm_values['attention1'].shape)
    # print(interm_values['attention2'].shape)
    # print(interm_values['attention3'].shape)