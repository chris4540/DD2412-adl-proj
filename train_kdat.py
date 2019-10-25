"""
Baseline algorithm to compare


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
import argparse
import os
from os.path import join
import math

class Config:
    """
    Static config
    """
    beta = 250
    input_shape = (32, 32, 3)
    batch_size = 128
    # We need to have 80k iterations for cifar 10
    epochs = math.ceil(80000 * batch_size / 50000)
    momentum = 0.9
    weight_decay = 5e-4
    init_lr = 0.1
    classes = 10

def lr_schedule(epoch):
    """
    Although we do not change parameters, hard coding is a very bad habbit.

    # of operations < 20 is negligible when we have 30s per epoch.
    """
    init_lr = Config.init_lr
    fraction = epoch / Config.epochs
    if fraction >= 0.8:
        return init_lr * (0.2**3)
    elif fraction >= 0.6:
        return init_lr * (0.2**2)
    elif fraction >= 0.3:
        return init_lr * 0.2
    return 0.1

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--tdepth', type=int, required=True)
    parser.add_argument('-tw', '--twidth', type=int, required=True)
    parser.add_argument('-sd', '--sdepth', type=int, required=True)
    parser.add_argument('-sw', '--swidth', type=int, required=True)
    parser.add_argument('-m', '--sample_per_class', type=int, required=True)
    parser.add_argument('-twghs','--teacher_weights', type=str, required=True,
                        help='Teacher weighting hdf5 file')
    parser.add_argument('--savedir', default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    return parser

def progressbar(it, prefix="", size=40, file=sys.stdout):
    """
    Progress bar showing
    Args:
        TODO
    Return:
        the yeild from the input iterator/generator

    See also:
        https://stackoverflow.com/a/34482761
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

# ============================================================================
# main
if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)

    # Print info
    print("-------------------------------------")
    print("Info:")
    # The name of this training
    train_name = "kdat-{dataset}_T{tdepth}-{twidth}_S{sdepth}-{swidth}_seed{seed}".format(**vars(args))
    print("Training name: ", train_name)

    # The save directory
    if args.savedir:
        savedir = args.savedir
    else:
        savedir = os.path.join(os.getcwd(), train_name)
    print("Save dir: ", savedir)

    # print out config
    for attr, v in vars(Config).items():
        if attr.startswith('__'):
            continue
        print(attr, ": ", v)
    print("-------------------------------------")

    # ===================================
    # Go to have training
    # load cifar 10, TODO: make a for SVHN
    x_train, y_train, x_test, y_test = preprocess.get_cifar_data()

    # load teacher
    teacher = WideResidualNetwork(
        args.tdepth, args.twidth, classes=Config.classes,
        input_shape=Config.input_shape,
        has_softmax=False, output_activations=True)
    # load from the hdf5 file. Use train_scratch to train it
    teacher.load_weights(args.teacher_weights)
    teacher.trainable = False

    # make student
    student = WideResidualNetwork(args.sdepth, args.sdepth, classes=Config.classes,
                                  input_shape=Config.input_shape,
                                  has_softmax=False, output_activations=True)
    # ==========================================================================
    # optimizer, like training from scratch
    optim = tf.keras.optimizers.SGD(learning_rate=lr_schedule(0),
                                    momentum=Config.momentum,
                                    nesterov=True)


    # Train student
    loss_metric = tf.keras.metrics.Mean()
    train_data_loader = tf.data.Dataset.from_tensor_slices(x_train).batch(128)
    for epoch in range(Config.epochs):
        print('Start of epoch {}'.format(epoch))

        # Iterate over the batches of the dataset.
        for x_batch_train in progressbar(train_data_loader, prefix="training"):
            step = 0
            # no checking on autodiff
            t_logits, *t_acts = teacher(x_batch_train)
            # Do forwarding, watch trainable varaibles and record auto grad.
            with tf.GradientTape() as tape:
                s_logits, *s_acts = student(x_batch_train)

                # The loss itself
                loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)
                # The L2 weighting regularization loss
                reg_loss = tf.reduce_sum(student.losses)

                # sum them up
                loss += Config.weight_decay * reg_loss

                grads = tape.gradient(loss, student.trainable_weights)
                optim.apply_gradients(zip(grads, student.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))
                step += 1
