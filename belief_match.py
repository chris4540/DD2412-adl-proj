"""
Refactor from ryan code
1. Sample only 1000 img for calculation
2. Use batch
"""
import tensorflow as tf
tf.enable_v2_behavior()
from net.wide_resnet import WideResidualNetwork
from utils.preprocess import get_cifar10_data
from utils.preprocess import balance_sampling
import numpy as np
from tqdm import tqdm

class Config:
    # K adversarial steps on network A
    adv_steps = 30
    batch_size = 100

    eta = 1.0
    n_classes = 10

def get_result(img_input, cls_pred, results):
    batch_size = tf.size(cls_i)
    # loop over different classes for perturb
    for cls_j in range(10):
        one_hot = tf.one_hot([cls_j]*batch_size, Config.n_classes)
        x_adv = tf.identity(img_input)
        for k in range(Config.adv_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                s_pred = student(x_adv)
                t_pred = teacher(x_adv)
                loss = cat_ce_fn(one_hot, s_pred)

            x_adv -= Config.eta*tape.gradient(loss, x_adv)
            # save down their predictions
            cls_prob_s = tf.reduce_max(s_pred * one_hot, axis=1)
            cls_prob_t = tf.reduce_max(t_pred * one_hot, axis=1)
            results[k].append((cls_prob_s, cls_prob_t))
    return results

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
    sel_x_test_list = []
    cls_pred_list = []
    offset = 0
    for batch_x, batch_y_lbl in test_data_loader:
        # Make prediction
        t_pred = tf.argmax(teacher(batch_x), -1)
        s_pred = tf.argmax(student(batch_x), -1)


        same_pred_idx = tf.compat.v2.where(tf.equal(t_pred, s_pred))
        # same_pred_idx = tf.cast(same_pred_idx, tf.int32)

        # select x_test and y_test only if two models pred. the same
        same_pred = tf.gather_nd(s_pred, same_pred_idx)
        sel_x_test = tf.gather_nd(x_test, same_pred_idx)
        cls_pred_list.append(same_pred)

        #
        sel_x_test_list.append(sel_x_test)


    cls_preds = tf.concat(cls_pred_list, 0)
    selected_x_test = tf.concat(sel_x_test_list, 0)

    n_match_data = tf.size(cls_preds).numpy()
    print("# of testing data for matching belief = ", n_match_data)
    # -----------------------------------------------------------------
    data = tf.data.Dataset.from_tensor_slices((selected_x_test, cls_preds)).batch(Config.batch_size)
    cat_ce_fn = tf.keras.losses.CategoricalCrossentropy()
    results = {k: [] for k in range(Config.adv_steps)}
    for x, cls_i in tqdm(data):
        results = get_result(x, cls_i, results)

    # ===========================================================

    # mean transition error
    err = 0
    for k in results.keys():
        diff = [tf.abs(a - b).numpy() for a, b in results[k]]
        [print(_.shape) for _ in diff]
        diff = np.concatenate(diff)
        err += np.mean(diff)
    print(err)

    # curve TODO
