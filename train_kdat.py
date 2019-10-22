"""

Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models

TODO:
    utils like accuracy etc.
"""
import tensorflow as tf
# Must run this in order to have similar result as TF2.0
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.wide_resnet import WideResidualNetwork
from utils import preprocess
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize

def lr_schedule(epoch):
    lr = 0.05
    if epoch > 160:
        lr *= 0.008
    elif epoch > 120:
        lr *= 0.04
    elif epoch > 60:
        lr *= 0.2
    print('Learning rate: ', lr)
    return lr

def spatial_attention_map(act_tensor, p=2):
    """
    Spatial attention mapping function to map the activation tensor with shape
    (H, W, C) to (H, W).

    We employed:
        sum of absolute values raised to the power of 2

    The f(A_{l}) is the paper of replication

    Args:
        act_tensor: activation tensor with shape (H, W, C)
    Output:
        a spatial attention map with shape (H, W)
    """

    out = tf.pow(act_tensor, p)
    out = tf.reduce_mean(out, axis=-1)
    # flatten it
    out = tf.reshape(out, [out.shape[0], -1])

    # renormalize them
    out = tf.linalg.l2_normalize(out)
    return out

def attention_loss(act1, act2):
    """
    Return the activation loss. The loss is the L2 distances between two
    activation map

    Args:
        act_map_1:
        act_map_2:

    Return:
        a floating point number representing the loss. As we use tensorflow,
        the floating point number would be a number hold in tf.Tensor

    TODO:
        check their implementation and code consistency

    Ref:
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L22
    """
    # get the activation map first
    act_map_1 = spatial_attention_map(act1)
    act_map_2 = spatial_attention_map(act2)

    # This is the author written in the paper
    # ret = tf.norm(act_map_2 - act_map_1, axis=-1)

    # This is the implementatin they have
    out = tf.pow(act_map_1 - act_map_2, 2)
    ret = tf.reduce_mean(out, axis=-1)
    return ret

# ============================================================================
# main
if __name__ == "__main__":
    beta = 250

    x_train, y_train, x_test, y_test = preprocess.get_cifar_data()

    teacher = WideResidualNetwork(
        40, 2, classes=10, input_shape=(32, 32, 3),
        has_softmax=False, output_activations=True)

    # load from the hdf5 file. Use train_scratch to train it
    teacher.load_weights('saved_models/cifar10_WRN-40-2_model.h5')

    teacher.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # student
    student = WideResidualNetwork(16, 1, classes=10, input_shape=(32, 32, 3),
                                  has_softmax=False, output_activations=True)
    # Train student
    # Iterate over epochs.
    kd_div = tf.keras.losses.KLD
    loss_metric = tf.keras.metrics.Mean()
    train_data_loader = tf.data.Dataset.from_tensor_slices(x_train).batch(128)

    for epoch in range(3):
        step = 0
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for x_batch_train in train_data_loader:
            # no checking on autodiff
            t_logits, t_act1, t_act2, t_act3 = teacher(x_batch_train)

            with tf.GradientTape() as tape:
                s_logits, s_act1, s_act2, s_act3 = student(x_batch_train)
                kd_loss = kd_div(
                    tf.math.softmax(t_logits),
                    tf.math.softmax(s_logits))
                attention_loss_sum = (attention_loss(t_act1, s_act1)
                                     + attention_loss(t_act2, s_act2)
                                     + attention_loss(t_act3, s_act3))
                loss = kd_loss + beta*attention_loss_sum

                grads = tape.gradient(loss, student.trainable_weights)
                optimizer.apply_gradients(zip(grads, student.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))
                step += 1
