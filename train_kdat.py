"""

Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models

TODO:
    - utils like accuracy etc.
    - code refactoring
"""
import tensorflow as tf
# Must run this in order to have similar behaviour as TF2.0
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.wide_resnet import WideResidualNetwork
from utils import preprocess
from utils.losses import student_loss_fn
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
            t_logits, *t_acts = teacher(x_batch_train)

            # Do forwarding, watch trainable varaibles and record auto grad.
            with tf.GradientTape() as tape:
                s_logits, *s_acts = student(x_batch_train)
                loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, beta)

                grads = tape.gradient(loss, student.trainable_weights)
                optimizer.apply_gradients(zip(grads, student.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))
                step += 1
