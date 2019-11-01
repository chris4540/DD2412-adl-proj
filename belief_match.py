"""
Refactor from ryan code
1. Sample only 1000 img for calculation
2. Use batch

Pros:
1. much faster

Cons:
1. The max prob. is not 1
2. Show only the relative different
"""
import tensorflow as tf
tf.enable_v2_behavior()
from net.wide_resnet import WideResidualNetwork
from utils.preprocess import get_cifar10_data
from utils.preprocess import balance_sampling
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import argparse

class Config:
    # K adversarial steps on network A
    adv_steps = 100
    data_per_class = 100
    eta = 1.0
    n_classes = 10

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', dest='output_csv', type=str, required=True)
    parser.add_argument('-t', '--teacher', dest='teacher_weight', type=str, required=True,
                        help='teacher weighting (network B in the paper)')
    parser.add_argument('-s', '--student', dest='student_weight', type=str, required=True,
                        help='student weighting (network A in the paper)')
    parser.add_argument('-m', dest='data_per_class', type=int, default=100)
    return parser

if __name__ == "__main__":
    #
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)
    # data
    (_, _), (x_test, y_test_labels) = get_cifar10_data()
    x_test, y_test_labels = balance_sampling(x_test, y_test_labels, data_per_class=args.data_per_class)

    # make sure that every batch is a class
    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test_labels)).batch(args.data_per_class)

    # Teacher
    teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3))
    teacher.load_weights(args.teacher_weight)

    # Student
    student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3))
    student.load_weights(args.student_weight)
    # student.load_weights('kdat-m200-cifar10_T40-2_S16-1_seed23_model.204.h5')

    # make them freeze
    student.trainable = False
    teacher.trainable = False
    # ========================================================
    # check if matched of two model predictions
    # use mini-batch for memory issue
    sel_x_test_list = []
    cls_pred_list = []
    # n_match_data = 0
    for batch_x, batch_y_lbl in test_data_loader:
        # Make prediction
        t_pred = tf.argmax(teacher(batch_x), -1)
        s_pred = tf.argmax(student(batch_x), -1)

        same_pred_idx = tf.compat.v2.where(tf.equal(t_pred, s_pred))

        # select x_test and y_test only if two models pred. the same
        same_pred = tf.gather_nd(s_pred, same_pred_idx)
        sel_x_test = tf.gather_nd(x_test, same_pred_idx)
        #
        cls_pred_list.append(same_pred)
        sel_x_test_list.append(sel_x_test)


    cls_preds = tf.concat(cls_pred_list, 0)
    selected_x_test = tf.concat(sel_x_test_list, 0)

    n_match_data = tf.size(cls_preds).numpy()
    print("# of testing data for matching belief = ", n_match_data)
    ind = np.argsort(cls_preds, axis=0)
    sorted_cls_preds = tf.gather(cls_preds, ind)
    sorted_imgs = tf.gather(selected_x_test, ind)

    # -----------------------------------------------------------------
    # split batches
    cur_cls = sorted_cls_preds[0].numpy()
    start_idx = 0
    img_batches = []
    classes = []
    for idx, cls_i in enumerate(sorted_cls_preds):
        if cur_cls != cls_i.numpy():
            # close this batch
            img_batches.append(sorted_imgs[start_idx:idx])
            classes.append(cur_cls)
            # Update
            start_idx = idx
            cur_cls = cls_i.numpy()
    # ------------------------------------------
    # the last class
    img_batches.append(sorted_imgs[start_idx:idx])
    classes.append(cls_i.numpy())
    # -----------------------------------------------------------------

    cat_ce_fn = tf.keras.losses.CategoricalCrossentropy()
    results = {k: [] for k in range(Config.adv_steps)}
    for batch_img, cls_i in tqdm(zip(img_batches, classes)):
        batch_size = int(batch_img.shape[0])
        # loop over different classes for perturb
        for cls_j in range(Config.n_classes):
            if cls_i == cls_j:
                continue
            one_hot = tf.one_hot([cls_j]*batch_size, Config.n_classes)
            x_adv = tf.identity(batch_img)
            for k in range(Config.adv_steps):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    s_pred = student(x_adv)
                    t_pred = teacher(x_adv)
                    loss = cat_ce_fn(one_hot, s_pred)

                x_adv -= Config.eta*batch_size*tape.gradient(loss, x_adv)
                # save down their predictions
                cls_prob_s = tf.reduce_max(s_pred * one_hot, axis=1)
                cls_prob_t = tf.reduce_max(t_pred * one_hot, axis=1)
                results[k].append((cls_prob_s, cls_prob_t))
    # ===========================================================
    # save down mean over classes and number of samples
    mean_t_prob_j = {k: None for k in range(Config.adv_steps)}
    mean_s_prob_j = {k: None for k in range(Config.adv_steps)}
    for k in results.keys():
        paris = results[k]
        s_probs = []
        t_probs = []
        for s, t in paris:
            s_probs.append(s)
            t_probs.append(t)

        s_probs = tf.concat(s_probs, 0)
        t_probs = tf.concat(t_probs, 0)

        mean_s_prob_j[k] = np.mean(s_probs.numpy())
        mean_t_prob_j[k] = np.mean(t_probs.numpy())
    # ==================================================================
    df = pd.DataFrame.from_dict({
        'teacher': mean_t_prob_j,
        'student': mean_s_prob_j,
    })

    df.to_csv(args.output_csv)

