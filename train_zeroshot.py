"""
Pesudo code for  Zero-shot KT

for 1..N; do  // outer loop

    * Sampling a seed `z`

    // loop for training generator to produce an adversarial example between teacher
    // and student.
    for 1..n_g; do
        * Generate an image from NavieGenerator
        * Update the parameters of NavieGenerator
            s.t. student and teacher has the largest discrepancy (in terms of KD-Div)
    done

    // Train student for the adversarial example
    for 1..n_s; do
        * Align teacher and student with standard KD-AT
    done
done

TODO:
    1. Add regularization_loss: https://stackoverflow.com/q/56693863
"""
import tensorflow as tf
tf.enable_v2_behavior()
import numpy as np
import os
import argparse
from tqdm import tqdm
import pprint
import time
import math
from collections import OrderedDict
from utils import mkdir
from utils.eval import evaluate
from os.path import join
from utils.seed import set_seed
from net.generator import NavieGenerator
from utils.losses import student_loss_fn
from utils.losses import generator_loss_fn
from utils.losses import knowledge_distil_loss_fn
from utils.preprocess import get_cifar10_data
from utils.preprocess import to_categorical
from utils.preprocess import balance_sampling
from utils.csvlogger import CustomizedCSVLogger
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class Config:
    """
    This config should be static as we follow the paper
    """
    # the random seed dimension
    z_dim = 100
    batch_size = 128
    # input shape
    input_dim = (32, 32, 3)
    # The number of inner loop to train generator to have an adversarial example
    n_g_in_loop = 1
    # The number of inner loop to align teacher and student with kd-at
    n_s_in_loop = 10
    # The weigting of the attention term
    beta = 250
    # The number of steps of the outer loop. The "N" in Algorithm 1
    n_outer_loop = 80000
    # n_outer_loop = 8000

    # clip grad
    clip_grad = 5.0

    # print freq
    print_freq = 1

    # log freq
    log_freq = 50

    # init learing rates
    student_init_lr = 2e-3
    generator_init_lr = 1e-3
    # ---------------------------
    alpha = 0.9
    temp = 4.0


def logits_to_distribution(logits):
    cls, cnt = np.unique(np.argmax(logits, axis=-1), return_counts=True)
    ret = dict(zip(cls, cnt))
    return ret


@tf.function
def train_gen(generator, g_optim, z_val, teacher, student):
    # ----------------------------------------------------------------
    with tf.GradientTape() as tape:
        pseudo_imgs = generator(z_val, training=True)
        t_logits, *t_acts = teacher(pseudo_imgs, training=False)
        s_logits, *_ = student(pseudo_imgs, training=True)
        # calculate the generator loss
        loss = generator_loss_fn(t_logits, s_logits)
    # ----------------------------------------------------------------

    # The grad for generator
    grads = tape.gradient(loss, generator.trainable_weights)

    # clip gradients to advoid large jump
    # g_grad_norm = 0
    grads, g_grad_norm = tf.clip_by_global_norm(grads, Config.clip_grad)

    # update the generator paramter with the gradient
    g_optim.apply_gradients(zip(grads, generator.trainable_weights))

    return loss, g_grad_norm


@tf.function
def train_student(pseudo_imgs, s_optim, t_logits, t_acts, student):

    # pseudo_imgs = generator(z_val, training=False)
    # t_logits, *t_acts = teacher(pseudo_imgs, training=False)
    # ----------------------------------------------------------------
    with tf.GradientTape() as tape:
        s_logits, *s_acts = student(pseudo_imgs, training=True)
        loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)
    # ----------------------------------------------------------------
    # The grad for student
    grads = tape.gradient(loss, student.trainable_weights)

    # clip gradients to advoid large jump
    grads, s_grad_norm = tf.clip_by_global_norm(grads, Config.clip_grad)


    # Apply grad for student
    s_optim.apply_gradients(zip(grads, student.trainable_weights))
    return loss, s_grad_norm, s_logits

@tf.function
def prepare_train_student(generator, z_val, teacher):
    pseudo_imgs = generator(z_val, training=True)
    t_logits, *t_acts = teacher(pseudo_imgs, training=False)
    return pseudo_imgs, t_logits, t_acts

@tf.function
def forward(model, batch, training):
    logits, *acts = model(batch, training=training)
    return logits, acts

@tf.function
def train_student_with_labels(student, optim, batch, t_logits, t_acts, onehot_label):
    # Do forwarding, watch trainable varaibles and record auto grad.
    with tf.GradientTape() as tape:
        s_logits, *s_acts = student(batch, training=True)
        # The loss itself
        loss = knowledge_distil_loss_fn(
                t_logits=t_logits,
                t_acts=t_acts,
                s_logits=s_logits,
                s_acts=s_acts,
                onehot_label=onehot_label,
                alpha=Config.alpha,
                beta=Config.beta,
                temp=Config.temp)
        # -------------------------------------------------
        # The L2 weighting regularization loss
        reg_loss = tf.reduce_sum(student.losses)

        # sum them up
        loss = loss + reg_loss
    grads = tape.gradient(loss, student.trainable_weights)
    optim.apply_gradients(zip(grads, student.trainable_weights))

    return loss

def zeroshot_train(t_depth, t_width, t_wght_path, s_depth, s_width,
                   seed=42, savedir=None, dataset='cifar10', sample_per_class=0):

    set_seed(seed)

    train_name = '%s_T-%d-%d_S-%d-%d_seed_%d' % (dataset, t_depth, t_width, s_depth, s_width, seed)
    log_filename = train_name + '_training_log.csv'

    # save dir
    if not savedir:
        savedir = 'zeroshot_' + train_name
    full_savedir = os.path.join(os.getcwd(), savedir)
    mkdir(full_savedir)

    log_filepath = os.path.join(full_savedir, log_filename)
    logger = CustomizedCSVLogger(log_filepath)

    # Teacher
    teacher = WideResidualNetwork(t_depth, t_width,
                                  input_shape=Config.input_dim,
                                  dropout_rate=0.0,
                                  output_activations=True,
                                  has_softmax=False)

    teacher.load_weights(t_wght_path)
    teacher.trainable = False

    # Student
    student = WideResidualNetwork(s_depth, s_width,
                                  input_shape=Config.input_dim,
                                  dropout_rate=0.0,
                                  output_activations=True,
                                  has_softmax=False)

    if sample_per_class > 0:
        s_decay_steps = Config.n_outer_loop*Config.n_s_in_loop + Config.n_outer_loop
    else:
        s_decay_steps = Config.n_outer_loop*Config.n_s_in_loop

    s_optim = Adam(learning_rate=CosineDecay(
                                Config.student_init_lr,
                                decay_steps=s_decay_steps))
    # ---------------------------------------------------------------------------
    # Generator
    generator = NavieGenerator(input_dim=Config.z_dim)
    g_optim = Adam(learning_rate=CosineDecay(
                                Config.generator_init_lr,
                                decay_steps=Config.n_outer_loop*Config.n_g_in_loop))
    # ---------------------------------------------------------------------------
    # Test data
    (x_train, y_train_lbl), (x_test, y_test) = get_cifar10_data()
    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)
    # ---------------------------------------------------------------------------
    # Train data (if using train data)
    train_dataflow = None
    if sample_per_class > 0:
        # sample first
        x_train, y_train_lbl = \
            balance_sampling(x_train, y_train_lbl, data_per_class=sample_per_class)
        datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4,
                                     horizontal_flip=True, vertical_flip=False,
                                     rescale=None, fill_mode='reflect')
        datagen.fit(x_train)
        y_train = to_categorical(y_train_lbl)
        train_dataflow = datagen.flow(x_train, y_train, batch_size=Config.batch_size, shuffle=True)


    # Generator loss metrics
    g_loss_met = tf.keras.metrics.Mean()

    # Student loss metrics
    s_loss_met = tf.keras.metrics.Mean()

    #
    n_cls_t_pred_metric = tf.keras.metrics.Mean()
    n_cls_s_pred_metric = tf.keras.metrics.Mean()

    max_g_grad_norm_metric = tf.keras.metrics.Mean()
    max_s_grad_norm_metric = tf.keras.metrics.Mean()


    teacher.trainable = False

    # checkpoint
    chkpt_dict = {
        'teacher': teacher,
        'student': student,
        'generator': generator,
        's_optim': s_optim,
        'g_optim': g_optim,
    }
    # Saving checkpoint
    ckpt = tf.train.Checkpoint(**chkpt_dict)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(savedir, 'chpt'), max_to_keep=2)
    # ==========================================================================
    # if a checkpoint exists, restore the latest checkpoint.
    start_iter = 0
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
        with open(os.path.join(savedir, 'chpt', 'iteration'), 'r') as f:
            start_iter = int(f.read())
        logger = CustomizedCSVLogger(log_filepath, append=True)

    for iter_ in range(start_iter, Config.n_outer_loop):
        iter_stime = time.time()

        max_s_grad_norm = 0
        max_g_grad_norm = 0
        # sample from latern space to have an image
        z_val = tf.random.normal([Config.batch_size, Config.z_dim])

        # Generator training
        loss = 0
        for ng in range(Config.n_g_in_loop):
            loss, g_grad_norm = train_gen(generator, g_optim, z_val, teacher, student)
            max_g_grad_norm = max(max_g_grad_norm, g_grad_norm.numpy())
            g_loss_met(loss)

        # ==========================================================================
        # Student training
        loss = 0
        pseudo_imgs, t_logits, t_acts = prepare_train_student(generator, z_val, teacher)
        for ns in range(Config.n_s_in_loop):
            # pseudo_imgs, t_logits, t_acts = prepare_train_student(generator, z_val, teacher)
            loss, s_grad_norm, s_logits = train_student(pseudo_imgs, s_optim, t_logits, t_acts, student)
            max_s_grad_norm = max(max_s_grad_norm, s_grad_norm.numpy())

            n_cls_t_pred = len(np.unique(np.argmax(t_logits, axis=-1)))
            n_cls_s_pred = len(np.unique(np.argmax(s_logits, axis=-1)))
            # logging
            s_loss_met(loss)
            n_cls_t_pred_metric(n_cls_t_pred)
            n_cls_s_pred_metric(n_cls_s_pred)
        # ==========================================================================
        # train if provided n samples
        if train_dataflow:
            x_batch_train, y_batch_train = next(train_dataflow)
            t_logits, t_acts = forward(teacher, x_batch_train, training=False)
            loss = train_student_with_labels(student, s_optim, x_batch_train, t_logits, t_acts, y_batch_train)
        # ==========================================================================

        # --------------------------------------------------------------------
        iter_etime = time.time()
        max_g_grad_norm_metric(max_g_grad_norm)
        max_s_grad_norm_metric(max_s_grad_norm)
        # --------------------------------------------------------------------
        is_last_epoch = (iter_ == Config.n_outer_loop - 1)

        if iter_!= 0 and (iter_ % Config.print_freq == 0 or is_last_epoch):
            n_cls_t_pred_avg = n_cls_t_pred_metric.result().numpy()
            n_cls_s_pred_avg = n_cls_s_pred_metric.result().numpy()
            time_per_epoch =  iter_etime - iter_stime

            s_loss = s_loss_met.result().numpy()
            g_loss = g_loss_met.result().numpy()
            max_g_grad_norm_avg = max_g_grad_norm_metric.result().numpy()
            max_s_grad_norm_avg = max_s_grad_norm_metric.result().numpy()

            # build ordered dict
            row_dict = OrderedDict()

            row_dict['time_per_epoch'] = time_per_epoch
            row_dict['epoch'] = iter_
            row_dict['generator_loss'] = g_loss
            row_dict['student_kd_loss'] = s_loss
            row_dict['n_cls_t_pred_avg'] = n_cls_t_pred_avg
            row_dict['n_cls_s_pred_avg'] = n_cls_s_pred_avg
            row_dict['max_g_grad_norm_avg'] = max_g_grad_norm_avg
            row_dict['max_s_grad_norm_avg'] = max_s_grad_norm_avg
            row_dict['s_optim_lr'] = s_optim.learning_rate(iter_*Config.n_s_in_loop).numpy()
            row_dict['g_optim_lr'] = g_optim.learning_rate(iter_).numpy()

            pprint.pprint(row_dict)
        # ======================================================================
        if iter_!= 0 and (iter_ % Config.log_freq == 0 or is_last_epoch):
            # calculate acc
            test_accuracy = evaluate(test_data_loader, student).numpy()
            row_dict['test_acc'] = test_accuracy
            logger.log_with_order(row_dict)
            print('Test Accuracy: ', test_accuracy)

            # for check poing
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                                                iter_+1, ckpt_save_path))
            with open(os.path.join(savedir, 'chpt', 'iteration'), 'w') as f:
                f.write(str(iter_+1))

            s_loss_met.reset_states()
            g_loss_met.reset_states()
            max_g_grad_norm_metric.reset_states()
            max_s_grad_norm_metric.reset_states()

        if iter_!= 0 and (iter_ % 5000 == 0 or is_last_epoch):
            generator.save_weights(join(full_savedir, "generator_i{}.h5".format(iter_)))
            student.save_weights(join(full_savedir, "student_i{}.h5".format(iter_)))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--tdepth', type=int, required=True)
    parser.add_argument('-tw', '--twidth', type=int, required=True)
    parser.add_argument('-sd', '--sdepth', type=int, required=True)
    parser.add_argument('-sw', '--swidth', type=int, required=True)
    parser.add_argument('-twgt','--teacher_weights', type=str, required=True)
    parser.add_argument('--savedir', default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('-m', '--sample_per_class', type=int, default=0)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    zeroshot_train(args.tdepth, args.twidth, args.teacher_weights,
                   args.sdepth, args.swidth,
                   seed=args.seed, savedir=args.savedir,
                   sample_per_class=args.sample_per_class)
