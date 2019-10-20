import os
import sys
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from models import *

def set_seed(seed):
    # NumPy
    import numpy as np
    np.random.seed(seed)
    # Python
    import random
    random.seed(seed)
#    from tensorflow import set_random_seed
#    set_random_seed(seed)


def get_cifar_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
    x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test

def lr_schedule(epoch):
    lr = 1e-1
    if epoch > 160:
        lr *= 0.008
    elif epoch > 120:
        lr *= 0.04
    elif epoch > 60:
        lr *= 0.2
    print('Learning rate: ', lr)
    return lr




depth = 16
width = 1
batch_size = 128
epochs = 200
seed = 42
model_type = 'WRN-%d-%d' % (depth, width)
shape, classes = (32, 32, 3), 10

set_seed(seed)
wrn_model = build_model(shape, classes, depth, width)

x_train, y_train, x_test, y_test = get_cifar_data()

# opt
opt = SGD(learning_rate=lr_schedule(0), momentum=0.9)
# opt = Adam(learning_rate=lr_schedule(0))

# compile model
wrn_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint, lr_scheduler]

datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

wrn_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1,
                        callbacks=callbacks)

scores = wrn_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
