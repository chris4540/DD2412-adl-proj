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



# def lr_schedule(epoch):
#     if epoch > 160:
#         print('lr: 0.0008')
#         return 0.0008
#     elif epoch > 120:
#         print('lr: 0.004')
#         return 0.004
#     elif epoch > 60:
#         print('lr: 0.02')
#         return 0.02
#     print('lr: 0.1')
#     return 0.1

# def random_pad_crop(image):
#     pad_size = 4

#     # padding to four edges
#     paddings = ([pad_size, pad_size], [pad_size, pad_size], [0, 0])
#     padded_img = np.pad(image, paddings, 'reflect')

#     # select the starting point
#     y = np.random.randint(0, 2*pad_size+1)
#     x = np.random.randint(0, 2*pad_size+1)
#     # set the size of cropped img, should be the same as input
#     dy, dx, _ = image.shape
#     ret = padded_img[y:(y+dy), x:(x+dx), :]
#     return ret

def get_piecewise_lr_schedule_fn():
    """
    The initial learning rate is set to 0.1 and is divided by 5 at 30%, 60%, and 80% of the run.

    Ref:
    https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay
    """
    boundaries = (Config.epochs * np.array([.3, .6, .8, 1]) - 1).astype(int)
    lr_values = Config.init_lr * np.array([.2**(i) for i in range(len(boundaries)+1)])
    # boundaries: [60, 121, 162, 203]
    # lr_values: [0.1    , 0.02   , 0.004  , 0.0008 , 0.00016]

    # convert them to list
    boundaries = boundaries.tolist()
    lr_values = lr_values.tolist()

    fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)
    return fn


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
    optim = SGD(learning_rate=get_piecewise_lr_schedule_fn(),
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
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpointer = ModelCheckpoint(filepath=model_filepath,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True
                                   )
    logger = CSVLogger(filename=log_filepath,
                       separator=',',
                       append=False
                       )

    callbacks = [checkpointer, logger]

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
    parser.add_argument('--seed', type=int, default=10)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args.depth, args.width, args.seed, savedir=args.savedir)
