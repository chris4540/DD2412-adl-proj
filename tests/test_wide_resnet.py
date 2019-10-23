"""
How to run:
    cd <project root>
    python -m unittest tests/test_get_outputs.py

Ref:
https://guillaumegenthial.github.io/testing.html
"""

import unittest
from net.wide_resnet import WideResidualNetwork
import tensorflow as tf
import numpy as np
import os.path

class Test(unittest.TestCase):
    input_shape = (32, 32, 3)
    depth = 10
    width = 1
    n_classes = 10
    weight_fname = 'test_wrn_tmp.h5'

    @classmethod
    def setUpClass(cls):
        cls.model_w_att = WideResidualNetwork(
                    cls.depth, cls.width,
                    classes=cls.n_classes,
                    input_shape=cls.input_shape,
                    has_softmax=False,
                    output_activations=True)

        cls.model_wo_att = WideResidualNetwork(
                    cls.depth, cls.width,
                    classes=cls.n_classes,
                    input_shape=cls.input_shape,
                    has_softmax=False,
                    output_activations=False)

        cls.model_out_prob = WideResidualNetwork(
                    cls.depth, cls.width,
                    classes=cls.n_classes,
                    input_shape=cls.input_shape)
    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.weight_fname):
            os.remove(cls.weight_fname)


    def test_output_shapes(self):
        batch_size = 128
        # create input values
        test_input = np.random.random([batch_size, *self.input_shape])
        with tf.Session():
            # convert it to tf
            tf_input = tf.constant(test_input, dtype=tf.float32)
            # try the model output
            logits, act1, act2, act3 = self.model_w_att(tf_input)

        self.assertEqual(logits.shape, (batch_size, self.n_classes))

        # attention1
        shape = (batch_size, 32, 32, 16*self.width)
        self.assertEqual(act1.shape, shape)

        # attention2
        shape = (batch_size, 16, 16, 32*self.width)
        self.assertEqual(act2.shape, shape)

        # attention3
        shape = (batch_size, 8, 8, 64*self.width)
        self.assertEqual(act3.shape, shape)

    def test_normal(self):

        batch_size = 128
        # create input values
        test_input = np.random.random([batch_size, *self.input_shape])

        with tf.Session():
            # convert it to tf
            tf_input = tf.constant(test_input, dtype=tf.float32)
            # try the model output
            prob = self.model_out_prob(tf_input)

        self.assertEqual(prob.shape, (batch_size, self.n_classes))

    def test_save_and_load(self):

        # model with attention => model without output attention
        self.model_w_att.save_weights(self.weight_fname)
        self.model_wo_att.load_weights(self.weight_fname)
        # ====================================================
