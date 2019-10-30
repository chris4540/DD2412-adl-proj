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
tf.enable_v2_behavior()
# Must run this in order to have similar behaviour as TF2.0
import argparse
import os
from os.path import join
import math
import sys
import utils
import time
from collections import OrderedDict
import numpy as np
from utils.seed import set_seed
from net.wide_resnet import WideResidualNetwork
from utils.eval import evaluate
from utils.preprocess import get_cifar10_data
from utils.preprocess import balance_sampling
from utils.preprocess import to_categorical
from utils.losses import student_loss_fn
from utils.losses import attention_loss
from utils.losses import kldiv_loss
from utils.csvlogger import CustomizedCSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Config:
    """
    Static config
    """
    beta = 250
    input_shape = (32, 32, 3)
    batch_size = 128
    # We need to have 80k iterations for cifar 10
    total_iteration = 80000
    momentum = 0.9
    weight_decay = 5e-4
    init_lr = 0.1
    classes = 10
    epochs = 205

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
    parser.add_argument('-twgt','--teacher_weights', type=str, required=True,
                        help='Teacher weighting hdf5 file')
    parser.add_argument('--savedir', default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    return parser

def cross_entropy(logits, onehot_label):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    ret = cce(onehot_label, logits)
    return ret

@tf.function
def forward(model, batch, training):
    logits, *acts = model(batch, training=training)
    return logits, acts

@tf.function
def train_student(student, optim, batch, t_logits, t_acts, onehot_label):
    # Do forwarding, watch trainable varaibles and record auto grad.
    with tf.GradientTape() as tape:
        s_logits, *s_acts = student(batch, training=True)
        # The loss itself
        loss = (2*0.9*1) * kldiv_loss(
                tf.math.softmax(t_logits / 1) ,
                tf.math.softmax(s_logits / 1))
        loss = loss + (1-0.9)*cross_entropy(s_logits, onehot_label)

        if Config.beta != 0.0:
            att_loss = 0.0
            for t_act, s_act in zip(t_acts, s_acts):
                att_loss += attention_loss(t_act, s_act)

            loss += Config.beta * att_loss
        # -------------------------------------------------
        # The L2 weighting regularization loss
        reg_loss = tf.reduce_sum(student.losses)

        # sum them up
        loss = loss + reg_loss

    grads = tape.gradient(loss, student.trainable_weights)
    optim.apply_gradients(zip(grads, student.trainable_weights))

    return loss


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

    # Set seed
    set_seed(args.seed)

    # ===================================
    # Go to have training
    # load cifar 10, sampling if need; TODO: make a for SVHN
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
    teacher_acc = evaluate(test_data_loader, teacher).numpy()
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
    train_dataset_flow = datagen.flow((x_train, y_train), batch_size=Config.batch_size, shuffle=True)


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
        for x_batch_train, y_batch_train in train_dataset_flow:

            # no checking on autodiff
            eval_stime = time.time()
            t_logits, t_acts = forward(teacher, x_batch_train, training=False)
            t_eval_time += time.time() - eval_stime

            train_stime = time.time()
            loss = train_student(student, optim, x_batch_train, t_logits, t_acts, y_batch_train)
            s_train_time += time.time() - train_stime

            loss_metric(loss)

            iter_ += 1
            if iter_ >= iter_per_epoch:
                break
            if iter_ % 100 == 0:
                print("iter: {}, Avg. Loss = {}".format(iter_,loss_metric.result().numpy()))
        # ----------------------------------------------------------------------
        epoch_loss = loss_metric.result().numpy()
        test_acc = evaluate(test_data_loader, student).numpy()

        row_dict = OrderedDict()
        row_dict['epoch'] = epoch
        row_dict['duration'] = time.time() - epoch_start_time
        row_dict['loss'] = epoch_loss
        row_dict['test_acc'] = test_acc
        row_dict['l_rate'] = lr
        row_dict['teacher_eval_time'] = t_eval_time
        row_dict['student_train_time'] = s_train_time

        print("Epoch {epoch}: duration = {duration}; l_rate = {l_rate}; "
              "Loss = {loss}, test_acc = {test_acc}".format(**row_dict))
        logging.log_with_order(row_dict)

        # reset metrics
        loss_metric.reset_states()
        # ------------------------------------------------------------------
        def save_model():
            # save down the model
            model_wght_file = train_name + "_model.{}.h5".format(epoch)
            print("Saving file: {} ....".format(model_wght_file))
            student.save_weights(os.path.join(savedir, model_wght_file))

        if (test_acc > best_acc) and epoch > 10:
            print("{} is better then {}".format(test_acc, best_acc))
            # update the best acc
            best_acc = test_acc

            save_model()
        elif epoch == Config.epochs -1:
            save_model()
