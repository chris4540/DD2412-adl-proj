"""
Baseline algorithm to compare
Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models
https://github.com/tensorflow/tensorflow/issues/30596
TODO:
    - utils like accuracy etc.
    - code refactoring
    - code tidy up
    - Consider to use tf.function
"""
import tensorflow as tf
# Must run this in order to have similar behaviour as TF2.0
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
import argparse
import os
# from tqdm import tqdm
from os.path import join
import math
import sys
import utils
import time
import numpy as np
from net.wide_resnet import WideResidualNetwork
from utils.preprocess import get_cifar10_data
from utils.preprocess import balance_sampling
from utils.preprocess import to_categorical
from utils.losses import student_loss_fn
from utils.csvlogger import CustomizedCSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize

class Config:
    """
    Static config
    """
    beta = 250
    input_shape = (32, 32, 3)
    batch_size = 128
    # We need to have 80k iterations for cifar 10
    epochs = 200  # easier for comparsion
    total_iteration = 80000
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

def evaluate(data_loader, model, output_logits=True, output_activations=True):
    total = 0
    correct = 0
    for inputs, labels in data_loader:
        if output_activations:
            out, *_ = model(inputs, training=False)
        else:
            out = model(inputs, training=False)

        prob = tf.math.softmax(out, axis=-1)

        pred = tf.argmax(prob, axis=-1)
        equality = tf.equal(pred, tf.reshape(labels, [-1]))
        correct += tf.reduce_sum(tf.cast(equality, tf.float32))
        total += equality.shape[0]

    ret = correct / tf.cast(total, tf.float32)
    return ret.numpy()

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
    train_name = "kdat-m{sample_per_class}-{dataset}_T{tdepth}-{twidth}_S{sdepth}-{swidth}_seed{seed}".format(**vars(args))
    print("Training name: ", train_name)

    # The save directory
    if args.savedir:
        savedir = args.savedir
    else:
        savedir = os.path.join(os.getcwd(), train_name)
    print("Save dir: ", savedir)
    utils.mkdir(savedir)

    #
    print('sample_per_class:', args.sample_per_class)

    # print out config
    for attr, v in vars(Config).items():
        if attr.startswith('__'):
            continue
        print(attr, ": ", v)

    # calculate iterations
    iter_per_epoch = math.ceil(Config.total_iteration / Config.epochs)
    print("Iteration per epoch: ", iter_per_epoch)
    print("-------------------------------------")


    # ===================================
    # Go to have training
    # load cifar 10, sampling if need
    # TODO: make a for SVHN
    (x_train, y_train_lbl), (x_test, y_test_lbl) = get_cifar10_data()
    if args.sample_per_class < 5000:
        x_train, y_train_lbl = balance_sampling(x_train, y_train_lbl, data_per_class=args.sample_per_class)

    # For evaluation
    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test_lbl)).batch(200)
    # y_test = to_categorical(y_test_lbl)
    y_train = to_categorical(y_train_lbl)

    # load teacher
    teacher = WideResidualNetwork(
        args.tdepth, args.twidth, classes=Config.classes,
        input_shape=Config.input_shape,
        has_softmax=False, output_activations=True)
    # load from the hdf5 file. Use train_scratch to train it
    teacher.load_weights(args.teacher_weights)
    teacher.trainable = False
    teacher_acc = evaluate(test_data_loader, teacher)
    print("teacher_acc = ", teacher_acc)

    # make student
    student = WideResidualNetwork(
                args.sdepth, args.swidth,
                classes=Config.classes,
                input_shape=Config.input_shape,
                has_softmax=False, output_activations=True, weight_decay=Config.weight_decay)

    # ==========================================================================
    # optimizer, like training from scratch
    optim = tf.keras.optimizers.SGD(learning_rate=lr_schedule(0),
                                    momentum=Config.momentum, nesterov=True)

    # logging dict
    logging = CustomizedCSVLogger(os.path.join(savedir, 'log_{}.csv'.format(train_name)))
    # Train student
    loss_metric = tf.keras.metrics.Mean()

    datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4,
                                     horizontal_flip=True, vertical_flip=False,
                                     rescale=None, fill_mode='reflect')
    train_dataset_flow = datagen.flow(x_train, batch_size=Config.batch_size, shuffle=True)


    best_acc = -np.inf
    for epoch in range(Config.epochs):
        # Iterate over the batches of the dataset.

        # start time
        epoch_start_time = time.time()
        s_train_time = 0
        t_eval_time = 0

        # iteration counter
        iter_ = 0

        # learning rate
        lr = lr_schedule(epoch)
        optim.learning_rate = lr
        for x_batch_train in train_dataset_flow:

            # no checking on autodiff
            eval_stime = time.time()
            t_logits, *t_acts = teacher(x_batch_train, training=False)
            t_eval_time += time.time() - eval_stime

            # Do forwarding, watch trainable varaibles and record auto grad.
            with tf.GradientTape() as tape:
                train_stime = time.time()
                s_logits, *s_acts = student(x_batch_train, training=True)

                # The loss itself
                loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)

                # The L2 weighting regularization loss
                reg_loss = tf.reduce_sum(student.losses)

                # sum them up
                loss = loss + reg_loss

                grads = tape.gradient(loss, student.trainable_weights)
                optim.apply_gradients(zip(grads, student.trainable_weights))

                s_train_time += time.time() - train_stime
                loss_metric(loss)

            iter_ += 1
            if iter_ >= iter_per_epoch:
                break
            if iter_ % 100 == 0:
                print("iter: {}, Avg. Loss = {}".format(iter_,loss_metric.result().numpy()))
        # ----------------------------------------------------------------------
        epoch_loss = loss_metric.result().numpy()
        test_acc = evaluate(test_data_loader, student)

        row_dict = {
            'duration': time.time() - epoch_start_time,
            'epoch': epoch,
            'loss': epoch_loss,
            'test_acc': test_acc,
            'l_rate': lr,
            'teacher_eval_time': t_eval_time,
            'student_train_time': s_train_time,
        }
        print("Epoch {epoch}: duration = {duration}; l_rate = {l_rate}; "
              "Loss = {loss}, test_acc = {test_acc}".format(**row_dict))
        logging.log(**row_dict)

        # reset metrics
        loss_metric.reset_states()

        # ------------------------------------------------------------------
        if (test_acc > best_acc) and epoch > 10:
            print("{} is better then {}".format(test_acc, best_acc))

            # save down the model
            model_wght_file = train_name + "_model.{}.h5".format(epoch)
            print("Saving file: {} ....".format(model_wght_file))
            student.save_weights(os.path.join(savedir, model_wght_file))

            # update the best acc
            best_acc = test_acc