"""
Run:
$ python3 train_scratch.py 40 2

TODO:
    1. Fix seeds
    2. Consider upgrading randomCrop (v2.0)
"""
import os
import sys
from utils.preprocess import load_cifar10_data
from utils.preprocess import to_categorical
from utils.seed import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from net.wide_resnet import WideResidualNetwork
import numpy as np

def lr_schedule(epoch):
    """
    """
    lr = 1e-1
    if epoch > 160:
        lr *= 0.008
    elif epoch > 120:
        lr *= 0.04
    elif epoch > 60:
        lr *= 0.2
    print('Learning rate: ', lr)
    return lr

def random_pad_crop(img):
    """
    How about
    tf.image.random_crop
    """
    pad=4
    paddings = ([pad,pad], [pad,pad], [0,0])
    img = np.pad(img, paddings, 'reflect')
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = 32, 32
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    copped_image = img[y:(y+dy), x:(x+dx), :]
    #print(copped_image.shape)
    return copped_image

class Config:
    """
    Static config
    """
    pass

def train(depth=16, width=1):
    # seed = 42
    batch_size = 128
    epochs = 200

    model_type = 'WRN-%d-%d' % (depth, width)
    shape = (32, 32, 3)
    classes = 10

    # set_seed(seed)
    wrn_model = WideResidualNetwork(depth, width, classes=classes, input_shape=shape)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()

    # To one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # compile model
    optim = SGD(learning_rate=lr_schedule(0), 
                momentum=0.9, 
                decay=0.0005
                )

    wrn_model.compile(loss='categorical_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
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
            )

    datagen.fit(x_train)

    wrn_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1,
                            callbacks=callbacks)

    scores = wrn_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    train(int(sys.argv[1]), int(sys.argv[2]))
