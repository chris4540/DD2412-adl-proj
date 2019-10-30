"""
Refactor from ryan code
"""
from utils.preprocess import get_cifar10_data
from net.wide_resnet import WideResidualNetwork

if __name__ == "__main__":
    # data
    (_, _), (x_test, y_test_labels) = get_cifar10_data()

    # Teacher
    teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3))
    teacher.load_weights('cifar10_WRN-40-2-seed45_model.172.h5')
    student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3))
    teacher.load_weights('cifar10_WRN-16-1-seed45_model.171.h5')

    # ========================================================
    student.trainable = False
    teacher.trainable = False
    # ========================================================

