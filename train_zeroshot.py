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
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from utils.seed import set_seed
from net.generator import NavieGenerator
from utils.losses import kd_loss
from utils.losses import student_loss_fn
from utils.preprocess import load_cifar10_data
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
from tensorflow.keras.experimental import CosineDecay
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

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

    t_depth = 40
    t_width = 2

    s_depth = 16
    s_width = 2


def mkdir(dirname):
    save_dir = os.path.join(os.getcwd(), dirname)
    os.makedirs(save_dir, exist_ok=True)


def zeroshot_train(t_depth, t_width, t_path, s_depth=16, s_width=1, seed=42, savedir='zeroshot', dataset='cifar10'):

    set_seed(seed)

    model_config = '%s_T-%d-%d_S-%d-%d_%d' % (dataset, t_depth, t_width, s_depth, s_width, seed)
    model_name = '%s_model.h5' % model_config
    log_filename = model_config+'_training_log.csv'
    
    save_dir = os.path.join(os.getcwd(), savedir)
    mkdir(save_dir)

    model_filepath = os.path.join(save_dir, model_name)
    log_filepath = os.path.join(save_dir, log_filename)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_filepath, 'a'))
    logger.info("Iteration,Generator_Loss,Student_Loss,Student_Test_Loss,Student_Test_Accuracy")

    ## Teacher
    teacher = WideResidualNetwork(t_depth, t_width, input_shape=Config.input_dim, dropout_rate=0.0, output_activations=True)
    teacher.load_weights(t_path)
    teacher.trainable = False

    ## Student
    student = WideResidualNetwork(s_depth, s_width, input_shape=Config.input_dim, dropout_rate=0.0, output_activations=True)
    student_optimizer = Adam(learning_rate=CosineDecay(
                                Config.student_init_lr,
                                decay_steps=Config.n_outer_loop*Config.n_g_in_loop))
    ## Generator
    generator = NavieGenerator(input_dim=Config.z_dim)
    ## TODO: double check the annuealing setting
    generator_optimizer = Adam(learning_rate=CosineDecay(
                                Config.generator_init_lr,
                                decay_steps=Config.n_outer_loop*Config.n_s_in_loop))

    # Generator loss metrics
    g_loss_met = tf.keras.metrics.Mean()
    # Student loss metrics
    stu_loss_met = tf.keras.metrics.Mean()

    #Test data
    (_, _), (x_test, y_test) = load_cifar10_data()


    for iter_ in range(Config.n_outer_loop):

        # sample from latern space to have an image
        z = tf.random.normal([Config.batch_size, Config.z_dim])

        # Generator training
        generator.trainable = True
        student.trainable = False
        for ng in range(Config.n_g_in_loop):
            with tf.GradientTape() as tape:
                pseudo_imgs = generator(z)
                t_logits, *_ = teacher(pseudo_imgs)
                s_logits, *_ = student(pseudo_imgs)

                # calculate the generator loss
                gen_loss = generator_loss_fn(t_logits, s_logits)

            # The grad for generator
            grads = tape.gradient(gen_loss, generator.trainable_weights)

            # update the generator paramter with the gradient
            generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

            g_loss_met(gen_loss)

            if iter_ % 50 == 0:
                print('step %s: generator mean loss = %s' % (iter_, g_loss_met.result().numpy()))
        # ==========================================================================

        # Student training
        generator.trainable = False
        student.trainable = True
        for ns in range(Config.n_s_in_loop):

            t_logits, *t_acts = teacher(pseudo_imgs)
            with tf.GradientTape() as tape:
                s_logits, *s_acts = student(pseudo_imgs)
                stu_loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)

            # The grad for student
            grads = tape.gradient(stu_loss, student.trainable_weights)

            # Apply grad for student
            student_optimizer.apply_gradients(zip(grads, student.trainable_weights))

            stu_loss_met(stu_loss)

            if iter_ % 50 == 0:
                print('step %s - %s: studnt mean loss = %s' % (iter_, ns, stu_loss_met.result().numpy()))

        if (iter_ + 1) % (Config.n_outer_loop/200) == 0:
            test_loss, test_accuracy = get_accuracy(student, x_test, y_test)
            logger.info(iter_,gen_loss.numpy(),stu_loss.numpy(), test_loss, test_accuracy)

    student.save(model_filepath)


def get_accuracy(student_model, s_depth, s_width, x_test, y_test):
    model = WideResidualNetwork(s_depth, s_width, input_shape=(32, 32, 3), dropout_rate=0.0)
    model.set_weights(student_model.get_weights()) 
    model.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    return loss, accuracy


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tw', '--twidth', type=int, required=True)
    parser.add_argument('-td', '--tdepth', type=int, required=True)
    parser.add_argument('-sw', '--swidth', type=int, required=True)
    parser.add_argument('-sd', '--sdepth', type=int, required=True)
    parser.add_argument('--tpath','--teacherpath', type=str, required=True)
    parser.add_argument('--savedir', type=str, default='zeroshot')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    zeroshot(args.tdepth, args.twidth, args.teacherpath, args.sdepth, args.swidth, args.seed, savedir=args.savedir)
