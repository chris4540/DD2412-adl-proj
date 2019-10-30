"""
Refactor from ryan code
"""
import tensorflow as tf
tf.enable_v2_behavior()
from net.wide_resnet import WideResidualNetwork
from utils.preprocess import get_cifar10_data

if __name__ == "__main__":
    # data
    (_, _), (x_test, y_test_labels) = get_cifar10_data()
    y_train = to_categorical(y_train_lbl)
    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test_lbl)).batch(200)

    # Teacher
    teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3))
    teacher.load_weights('cifar10_WRN-40-2-seed45_model.172.h5')

    # Student
    student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3))
    student.load_weights('cifar10_WRN-16-1-seed45_model.171.h5')

    # make them freeze
    student.trainable = False
    teacher.trainable = False
    # ========================================================
    # ========================================================
    # Make prediction
    t_pred = tf.argmax(teacher(x_test), -1)
    s_pred = tf.argmax(student(x_test), -1)

    # check if matched
    common_pred = tf.reshape(tf.where(s_pred == t_pred), [-1])
