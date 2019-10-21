"""

Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""

from net.wide_resnet import WideResidualNetwork
from utils import preprocess
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
# ============================================================================
# main
if __name__ == "__main__":
    # =======================================================================
    teacher = WideResidualNetwork(40, 2, classes=10, input_shape=(32, 32, 3))

    x_train, y_train, x_test, y_test = preprocess.get_cifar_data()
    # compile model
    optim = SGD(learning_rate=lr_schedule(0), momentum=0.9, decay=0.0005)
    teacher.compile(loss='categorical_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)

    callbacks = [lr_scheduler]

    # use the plain generator
    datagen = ImageDataGenerator()

    datagen.fit(x_train)

    teacher.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                            validation_data=(x_test, y_test),
                            epochs=5, verbose=1,
                            callbacks=callbacks)

    scores = teacher.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # ====================================================================