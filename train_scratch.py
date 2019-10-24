"""
Run:
$ python3 train_scratch.py 40 2

TODO:
    1. Fix seeds
    2. Consider upgrading randomCrop (v2.0)
"""
import os
import sys
import math
import numpy as np
import tensorflow as tf
from utils.preprocess import load_cifar10_data
from utils.preprocess import to_categorical
from utils.seed import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from net.wide_resnet import WideResidualNetwork
from net.wide_resnet_v2 import build_model
import argparse

class Config:
    """
    Static config
    """
    batch_size = 128
    # We need to have 80k iterations for cifar 10
    epochs = 205
    momentum = 0.9
    weight_decay = 5e-4
    init_lr = 0.1



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


def train(depth, width, seed=42, dataset='cifar10', savedir='saved_models'):

    set_seed(seed)

    # Load data
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
        shape = (32, 32, 3)
        classes = 10
    else:
        raise NotImplementedError("TODO: SVHN")

    # Setup model
    model_type = 'WRN-%d-%d' % (depth, width)
    wrn_model = WideResidualNetwork(depth, width, classes=classes, input_shape=shape)
    #  wrn_model = build_model(shape, classes, depth, width)

    # To one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # compile model
    optim = SGD(learning_rate=lr_schedule(0),
                momentum=Config.momentum,
                decay=Config.weight_decay
                )

    wrn_model.compile(loss='categorical_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), savedir)
    mkdir(save_dir)

    # Set up model name and path
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    model_filepath = os.path.join(save_dir, model_name)
    log_filepath = os.path.join(save_dir, 'log.csv')

    # Prepare callbacks for model saving and for learning rate adjustment.
    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpointer = ModelCheckpoint(filepath=model_filepath,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True
                                   )
    logger = CSVLogger(filename=log_filepath,
                       separator=',',
                       append=False
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
        epochs=Config.epochs, verbose=1,
        callbacks=callbacks)

    scores = wrn_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=int, required=True)
    parser.add_argument('-d', '--depth', type=int, required=True)
    parser.add_argument('--savedir', type=str, default='savedir')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args.depth, args.width, args.seed, savedir=args.savedir)
