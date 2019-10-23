import unittest
import tensorflow as tf
from utils.preprocess import balance_sampling
import numpy as np

class TestBalanceSampler(unittest.TestCase):

    def test_use_case(self):
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        _, y_sample = balance_sampling(x_train, y_train, data_per_class=200)

        classes, counts = np.unique(y_sample, return_counts=True)

        self.assertEqual(len(classes), 10)

        for c in counts:
            self.assertEqual(c, 200)
