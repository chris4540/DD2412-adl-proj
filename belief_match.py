"""
Refactor from ryan code
"""
import tensorflow as tf
tf.enable_v2_behavior()
from net.wide_resnet import WideResidualNetwork
from utils.preprocess import get_cifar10_data
from utils.preprocess import balance_sampling

if __name__ == "__main__":
    # data
    (_, _), (x_test, y_test_labels) = get_cifar10_data()
    if True:
        x_test, y_test_labels = balance_sampling(x_test, y_test_labels, data_per_class=50)
    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test_labels)).batch(200)

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
    # check if matched of two model predictions
    # use mini-batch for memory issue
    same_pred_idx_list = []
    offset = 0
    for batch_x, batch_y_lbl in test_data_loader:
        # Make prediction
        t_pred = tf.argmax(teacher(batch_x), -1)
        s_pred = tf.argmax(student(batch_x), -1)


        same_pred = tf.compat.v2.where(tf.equal(t_pred, s_pred))
        same_pred = tf.cast(same_pred, tf.int32)
        same_pred = tf.reshape(same_pred, [-1]) + offset

        # add back offset
        offset += tf.size(batch_y_lbl)

        same_pred_idx_list.append(same_pred)

    same_pred_idxs = tf.concat(same_pred_idx_list, 0)
    n_match_data = tf.size(same_pred_idxs).numpy()
    print("# of testing data for matching belief = ", n_match_data)
    # --------------------------------------------------------
