"""
1. Used skills not mentioned in zero shot paper:
    1. random crop, fill_mode='reflect'
    2. horizontal_flip
    3. nesterov momentum
"""
import os
import sys
import math
import numpy as np
import tensorflow as tf
from utils.preprocess import get_cifar10_data
from utils.preprocess import get_fashion_mnist_data
from utils.preprocess import to_categorical
from utils.preprocess import balance_sampling
from utils.seed import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from net.wide_resnet import WideResidualNetwork
import argparse
from tensorflow.keras.models import load_model

class Config:
    """
    Static config
    """
    batch_size = 128
    # We need to have 80k iterations for cifar 10
    epochs = 205 # math.ceil(80000 * batch_size / 50000)
    momentum = 0.9
    weight_decay = 5e-4
    init_lr = 0.1
    n_data_per_class = 5000



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

def mkdir(dirname):
    save_dir = os.path.join(os.getcwd(), dirname)
    os.makedirs(save_dir, exist_ok=True)


def train(depth, width, seed=42, data_per_class=-1, dataset='cifar10', savedir='saved_models', is_continue=False):

    set_seed(seed)

    # Load data
    if dataset == 'cifar10':
        # TODO: sampling for Fig2 green line
        (x_train, y_train_lbl), (x_test, y_test_lbl) = get_cifar10_data()
        # x_train, y_train_lbl = balance_sampling(x_train, y_train_lbl, data_per_class=200)
        shape = (32, 32, 3)
        classes = 10
    elif dataset == 'fashion_mnist':
        (x_train, y_train_lbl), (x_test, y_test_lbl) = get_fashion_mnist_data()
        shape = (32, 32, 1)
        classes = 10
    else:
        raise NotImplementedError("TODO: SVHN")
    # ====================================================================
    # make sampling
    if data_per_class > 0:
        # sample
        x_train_sample, y_train_lbl_sample = \
            balance_sampling(x_train, y_train_lbl, data_per_class=data_per_class)

        # repeat the sampled data to be as large as the full data set for convienient
        x_train = np.repeat(x_train_sample, Config.n_data_per_class/data_per_class, axis=0)
        y_train_lbl = np.repeat(y_train_lbl_sample, Config.n_data_per_class/data_per_class, axis=0)
    # ====================================================================
    # To one-hot
    y_train = to_categorical(y_train_lbl)
    y_test = to_categorical(y_test_lbl)

    # Setup model
    model_type = 'WRN-%d-%d-seed%d' % (depth, width, seed)
    wrn_model = WideResidualNetwork(
            depth, width, classes=classes, input_shape=shape,
            weight_decay=Config.weight_decay)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), savedir)
    mkdir(save_dir)

    # Set up model name and path
    model_name = '%s_%s_model.{epoch:03d}.h5' % (dataset, model_type)
    model_filepath = os.path.join(save_dir, model_name)

    # set up log file
    log_fname = '{}-wrn-{}-{}-seed{}_log.csv'.format(dataset, depth, width, seed)
    log_filepath = os.path.join(save_dir, log_fname)
    # =================================================================
    if is_continue:
        for i in range(Config.epochs, 0, -1):
            fname = model_filepath.format(epoch=i)
            if os.path.isfile(fname):
                print("Using ", fname, " as the save point.")
                break
        if i <= 1:
            raise RuntimeError("Cannot continue the training")
        # ======================================================
        initial_epoch = i
        wrn_model = load_model(fname)
        is_log_append = True
    else:
        initial_epoch = 0
        # compile model
        optim = SGD(learning_rate=lr_schedule(initial_epoch),
                    momentum=Config.momentum,
                    decay=0.0,
                    nesterov=True
                    )

        wrn_model.compile(loss='categorical_crossentropy',
                        optimizer=optim,
                        metrics=['accuracy'])
        is_log_append = False

    logger = CSVLogger(filename=log_filepath, separator=',', append=is_log_append)

    # Prepare callbacks for model saving and for learning rate adjustment.
    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpointer = ModelCheckpoint(filepath=model_filepath,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True
                                   )

    callbacks = [lr_scheduler, checkpointer, logger]

    datagen = ImageDataGenerator(
            width_shift_range=4,
            height_shift_range=4,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            fill_mode='reflect',
            )

    datagen.fit(x_train)

    wrn_model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=Config.batch_size, shuffle=True),
        validation_data=(x_test, y_test),
        epochs=Config.epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=callbacks)

    scores = wrn_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # =================================================
    # use the final one as teachers
    wrn_model.save(model_filepath.format(epoch=Config.epochs-1))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=int, required=True)
    parser.add_argument('-d', '--depth', type=int, required=True)
    parser.add_argument('-m', '--sample_per_class', type=int, default=-1)
    parser.add_argument('--savedir', type=str, default='savedir')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--continue', dest='cont', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args.depth, args.width, seed=args.seed,
          data_per_class=args.sample_per_class,
          savedir=args.savedir, is_continue=args.cont)
