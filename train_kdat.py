"""

Ref:
https://github.com/keras-team/keras/issues/9459#issuecomment-469282443
https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""
import tensorflow as tf
# Must run this in order to have similar result as TF2.0
tf.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.wide_resnet import WideResidualNetwork
from utils import preprocess
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model


def lr_schedule(epoch):
    lr = 0.05
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
                            epochs=10, verbose=1,
                            callbacks=callbacks)

    scores = teacher.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # ====================================================================
    # re-create a model
    teacher = Model(teacher.input, teacher.get_layer('logits').output)
    # now model outputs logits
    print(teacher.summary())

    teacher.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # student
    student = WideResidualNetwork(16, 1, classes=10, input_shape=(32, 32, 3), has_softmax=False)
    # Train student
    # Iterate over epochs.
    kd_div = tf.keras.losses.KLD
    loss_metric = tf.keras.metrics.Mean()
    for epoch in range(3):
        step = 0
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for x_batch_train in datagen.flow(x_train, batch_size=128):
            # no checking on autodiff
            teacher_logits = teacher(x_batch_train)

            with tf.GradientTape() as tape:
                student_logits = student(x_batch_train)
                kd_loss = kd_div(teacher_logits, student_logits)
                loss = kd_loss + 0

                grads = tape.gradient(loss, student.trainable_weights)
                optimizer.apply_gradients(zip(grads, student.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))


    # # prepare logits

    # # Training
    # train_logits = []
    # for x_batch, _ in datagen.flow(x_train, batch_size=128):
    #     logits = teacher.predict_on_batch(x_batch)
    #     train_logits.append(logits)

    # # Validation
    # val_logits = []
    # for x_batch, _ in datagen.flow(x_test, batch_size=128):
    #     logits = teacher.predict_on_batch(x_batch)
    #     val_logits.append(logits)
