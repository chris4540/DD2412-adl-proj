import numpy as np
import random
import tensorflow as tf

def set_seed(seed):
    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)

    # Tensorflow
    tf.set_random_seed(seed)
