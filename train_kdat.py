"""

Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models

TODO:
    - utils like accuracy etc.
    - code refactoring

Sample outputs:

step 0: mean loss = tf.Tensor(2.316777, shape=(), dtype=float32)
step 100: mean loss = tf.Tensor(1.7968416, shape=(), dtype=float32)
step 200: mean loss = tf.Tensor(1.6433067, shape=(), dtype=float32)
step 300: mean loss = tf.Tensor(1.5375729, shape=(), dtype=float32)
Start of epoch 1
step 0: mean loss = tf.Tensor(1.4717181, shape=(), dtype=float32)
step 100: mean loss = tf.Tensor(1.4091233, shape=(), dtype=float32)
step 200: mean loss = tf.Tensor(1.3524677, shape=(), dtype=float32)
step 300: mean loss = tf.Tensor(1.305955, shape=(), dtype=float32)
Start of epoch 2
step 0: mean loss = tf.Tensor(1.266997, shape=(), dtype=float32)
step 100: mean loss = tf.Tensor(1.2284571, shape=(), dtype=float32)
step 200: mean loss = tf.Tensor(1.1918056, shape=(), dtype=float32)
step 300: mean loss = tf.Tensor(1.159136, shape=(), dtype=float32)
"""
import tensorflow as tf
# Must run this in order to have similar behaviour as TF2.0
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.wide_resnet import WideResidualNetwork
from utils import preprocess
from utils.losses import attention_loss
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
