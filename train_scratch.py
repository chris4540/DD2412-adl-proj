"""
Run:
$ python3 train_scratch.py 40 2

TODO:
    1. Fix seeds
    2. Consider upgrading randomCrop (v2.0)
"""
import tensorflow as tf
import os
import sys
from utils.preprocess import load_cifar10_data
from utils.preprocess import to_categorical
from utils.seed import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from net.wide_resnet import WideResidualNetwork
from net.wide_resnet_v2 import build_model
import numpy as np
import argparse

class Config:
    """
    Static config
    """
    batch_size = 128
    epochs = 204
    momentum = 0.9
    weight_decay = 5e-4

def lr_schedule(epoch):
    if epoch > 160:
        print('lr: 0.0008')
        return 0.0008
    elif epoch > 120:
        print('lr: 0.004')
        return 0.004
    elif epoch > 60:
        print('lr: 0.02')
        return 0.02
    print('lr: 0.1')
    return 0.1

def random_pad_crop(image):
    pad_size = 4

    # padding to four edges
    paddings = ([pad_size, pad_size], [pad_size, pad_size], [0, 0])
    padded_img = np.pad(image, paddings, 'reflect')

    # select the starting point
    y = np.random.randint(0, 2*pad_size+1)
    x = np.random.randint(0, 2*pad_size+1)
    # set the size of cropped img, should be the same as input
    dy, dx, _ = image.shape
    ret = padded_img[y:(y+dy), x:(x+dx), :]
    return ret

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
    log_filepath = os.path.join(save_dir, 'log.txt')

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
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            preprocessing_function=random_pad_crop,
            rescale=None,
            shear_range=10,
            zca_whitening=True
            )

    datagen.fit(x_train)

    wrn_model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=Config.batch_size),
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
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args.depth, args.width)
