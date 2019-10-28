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
# tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
tf.enable_v2_behavior()
from utils.seed import set_seed
from net.generator import NavieGenerator
from utils.losses import student_loss_fn, generator_loss_fn
from utils.preprocess import get_cifar10_data
from utils.csvlogger import CustomizedCSVLogger
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
from tensorflow.keras.experimental import CosineDecay
import numpy as np
import os
import argparse
from tqdm import tqdm
import pprint
import time
import math
from collections import OrderedDict
# TODO: use Config class
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

    # init learing rates
    student_init_lr = 2e-3
    generator_init_lr = 1e-3

    # generator
    # save_models_at = 800

    #weight_decay = 5e-4

def logits_to_distribution(logits):
    cls, cnt = np.unique(np.argmax(logits, axis=-1), return_counts=True)
    ret = dict(zip(cls, cnt))
    return ret

def mkdir(dirname):
    save_dir = os.path.join(os.getcwd(), dirname)
    os.makedirs(save_dir, exist_ok=True)

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
    grads, g_grad_norm = tf.clip_by_global_norm(grads, 5.0)

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
    grads, s_grad_norm = tf.clip_by_global_norm(grads, 5.0)
    # max_s_grad_norm = max(max_s_grad_norm, s_grad_norm.numpy())

    # Apply grad for student
    s_optim.apply_gradients(zip(grads, student.trainable_weights))
    return loss, s_grad_norm, s_logits

@tf.function
def prepare_train_student(generator, z_val, teacher):
    pseudo_imgs = generator(z_val, training=True)
    t_logits, *t_acts = teacher(pseudo_imgs, training=False)
    return pseudo_imgs, t_logits, t_acts


def zeroshot_train(t_depth, t_width, teacher_weights, s_depth=16, s_width=1,
                   seed=42, savedir=None, dataset='cifar10'):

    set_seed(seed)

    train_name = '%s_T-%d-%d_S-%d-%d_seed_%d' % (dataset, t_depth, t_width, s_depth, s_width, seed)
    log_filename = train_name + '_training_log.csv'

    if not savedir:
        savedir = 'zeroshot' + train_name
    save_dir = os.path.join(os.getcwd(), savedir)
    mkdir(save_dir)

    #model_filepath = os.path.join(save_dir, model_name)
    log_filepath = os.path.join(save_dir, log_filename)
    logger = CustomizedCSVLogger(log_filepath)

    ## Teacher
    teacher = WideResidualNetwork(t_depth, t_width,
                                  input_shape=Config.input_dim,
                                  dropout_rate=0.0,
                                  output_activations=True,
                                  has_softmax=False)

    teacher.load_weights(t_path)
    teacher.trainable = False

    ## Student
    student = WideResidualNetwork(s_depth, s_width,
                                  input_shape=Config.input_dim,
                                  dropout_rate=0.0,
                                  output_activations=True,
                                  has_softmax=False)

    s_optim = Adam(learning_rate=CosineDecay(
                                Config.student_init_lr,
                                decay_steps=Config.n_outer_loop*Config.n_s_in_loop))
    ## Generator
    generator = NavieGenerator(input_dim=Config.z_dim)
    ## TODO: double check the annuealing setting
    g_optim = Adam(learning_rate=CosineDecay(
                                Config.generator_init_lr,
                                decay_steps=Config.n_outer_loop*Config.n_g_in_loop))

    # Generator loss metrics
    g_loss_met = tf.keras.metrics.Mean()

    # Student loss metrics
    s_loss_met = tf.keras.metrics.Mean()

    #
    n_cls_t_pred_metric = tf.keras.metrics.Mean()
    n_cls_s_pred_metric = tf.keras.metrics.Mean()

    #Test data
    (_, _), (x_test, y_test) = get_cifar10_data()

    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

    teacher.trainable = False

    # checkpoint
    chkpt_dict = {
        'teacher': teacher,
        'student': student,
        'generator': generator,
        's_optim': s_optim,
        'g_optim': g_optim
    }
    # Saving checkpoint
    ckpt = tf.train.Checkpoint(**chkpt_dict)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(savedir, 'chpt'), max_to_keep=2)
    # ==========================================================================

    # for iter_ in tqdm(range(Config.n_outer_loop), desc="Global Training Loop"):
    for iter_ in range(Config.n_outer_loop):
        iter_stime = time.time()

        # max_s_grad_norm = 0
        # max_g_grad_norm = 0
        # sample from latern space to have an image
        z_val = tf.function(tf.random.normal)([Config.batch_size, Config.z_dim])

        # Generator training
        loss = 0
        for ng in range(Config.n_g_in_loop):
            loss, g_grad_norm = train_gen(generator, g_optim, z_val, teacher, student)
            # max_g_grad_norm = max(max_g_grad_norm, g_grad_norm.numpy())
            g_loss_met(loss)

        # ==========================================================================

        # Student training
        loss = 0
        pseudo_imgs, t_logits, t_acts = prepare_train_student(generator, z_val, teacher)
        for ns in range(Config.n_s_in_loop):
            loss, s_grad_norm, s_logits = train_student(pseudo_imgs, s_optim, t_logits, t_acts, student)
            # max_s_grad_norm = max(max_s_grad_norm, s_grad_norm.numpy())

            n_cls_t_pred = len(np.unique(np.argmax(t_logits, axis=-1)))
            n_cls_s_pred = len(np.unique(np.argmax(s_logits, axis=-1)))
            # logging
            s_loss_met(loss)
            n_cls_t_pred_metric(n_cls_t_pred)
            n_cls_s_pred_metric(n_cls_s_pred)
        # --------------------------------------------------------------------
        iter_etime = time.time()

        if iter_ % 100 == 0:
            n_cls_t_pred_avg = n_cls_t_pred_metric.result().numpy()
            n_cls_s_pred_avg = n_cls_s_pred_metric.result().numpy()
            time_per_epoch =  iter_etime - iter_stime

            s_loss = s_loss_met.result().numpy()
            g_loss = g_loss_met.result().numpy()

            # build ordered dict
            row_dict = OrderedDict()

            row_dict['time_per_epoch'] = time_per_epoch
            row_dict['epoch'] = iter_
            row_dict['generator_loss'] = g_loss
            row_dict['student_kd_loss'] = s_loss
            row_dict['n_cls_t_pred_avg'] = n_cls_t_pred_avg
            row_dict['n_cls_s_pred_avg'] = n_cls_s_pred_avg
            row_dict['s_optim_lr'] = s_optim.learning_rate(iter_*Config.n_s_in_loop).numpy()
            row_dict['g_optim_lr'] = g_optim.learning_rate(iter_).numpy()

            pprint.pprint(row_dict)
        # ======================================================================
        if iter_!= 0 and iter_ % 500 == 0:
            # calculate acc
            test_accuracy = evaluate(test_data_loader, student).numpy()
            row_dict['test_acc'] = test_accuracy
            logger.log_with_order(row_dict)
            print('Test Accuracy: ', test_accuracy)

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                                                iter_+1, ckpt_save_path))

        s_loss_met.reset_states()
        g_loss_met.reset_states()

def evaluate(data_loader, model, output_activations=True):
    total = 0
    correct = 0.0
    for inputs, labels in tqdm(data_loader):
        if output_activations:
            out, *_ = model(inputs, training=False)
        else:
            out = model(inputs, training=False)

        prob = tf.math.softmax(out, axis=-1)

        pred = tf.argmax(prob, axis=-1)
        equality = tf.equal(pred, tf.reshape(labels, [-1]))
        correct = correct + tf.reduce_sum(tf.cast(equality, tf.float32))
        total = total + tf.size(equality)

    total = tf.cast(total, tf.float32)
    ret = correct / total
    return ret


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
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    zeroshot_train(args.tdepth, args.twidth, args.teacher_weights, args.sdepth, args.swidth, args.seed, savedir=args.savedir)
