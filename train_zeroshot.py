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
    save_models_at = 800

    #weight_decay = 5e-4


def mkdir(dirname):
    save_dir = os.path.join(os.getcwd(), dirname)
    os.makedirs(save_dir, exist_ok=True)


def zeroshot_train(t_depth, t_width, t_path, s_depth=16, s_width=1, seed=42, savedir='zeroshot', dataset='cifar10'):

    #set_seed(seed)

    model_config = '%s_T-%d-%d_S-%d-%d_seed_%d' % (dataset, t_depth, t_width, s_depth, s_width, seed)
    #model_name = '%s_model.h5' % model_config
    log_filename = model_config + '_training_log.csv'

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

    # student_optimizer = Adam(learning_rate=CosineDecay(
    #                             Config.student_init_lr,
    #                             decay_steps=Config.n_outer_loop*Config.n_s_in_loop))

    student_optimizer = Adam(student_init_lr)
    ## Generator
    generator = NavieGenerator(input_dim=Config.z_dim)
    ## TODO: double check the annuealing setting
    # generator_optimizer = Adam(learning_rate=CosineDecay(
    #                             Config.generator_init_lr,
    #                             decay_steps=Config.n_outer_loop*Config.n_g_in_loop))

    generator_optimizer = Adam(generator_init_lr)

    # Generator loss metrics
    g_loss_met = tf.keras.metrics.Mean()
    # Student loss metrics
    stu_loss_met = tf.keras.metrics.Mean()

    #Test data
    (_, _), (x_test, y_test) = get_cifar10_data()

    test_data_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)


    for iter_ in tqdm(range(Config.n_outer_loop), desc="Global Training Loop"):

        # sample from latern space to have an image
        z = tf.random.normal([Config.batch_size, Config.z_dim])

        # Generator training
        #generator.trainable = True
        #student.trainable = False
        for ng in range(Config.n_g_in_loop):
            with tf.GradientTape() as gtape:
                pseudo_imgs = generator(z)
                t_logits, *t_acts = teacher(pseudo_imgs)
                s_logits, *_ = student(pseudo_imgs)

                # calculate the generator loss
                gen_loss = generator_loss_fn(t_logits, s_logits)
                # The grad for generator
                grads = gtape.gradient(gen_loss, generator.trainable_weights)
                # clip gradients
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                # update the generator paramter with the gradient
                generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

                g_loss_met(gen_loss)
                    

        # ==========================================================================

        # Student training
        #generator.trainable = False
        #student.trainable = True
        for ns in range(Config.n_s_in_loop):

            #t_logits, *t_acts = teacher(pseudo_imgs)
            with tf.GradientTape() as stape:
                #pseudo_imgs = generator(z)
                #t_logits, *t_acts = teacher(pseudo_imgs)
                s_logits, *s_acts = student(pseudo_imgs)
                stu_loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)

                # The grad for student
                grads = stape.gradient(stu_loss, student.trainable_weights)
                # clip gradients
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                # Apply grad for student
                student_optimizer.apply_gradients(zip(grads, student.trainable_weights))

                stu_loss_met(stu_loss)

        s_loss = stu_loss_met.result().numpy()
        g_loss = g_loss_met.result().numpy()

        if iter_ % 5 == 0:
            print('step %s | generator mean loss = %s | studnt mean loss = %s' % (iter_, g_loss, s_loss))

        if (iter_ + 1) % (Config.n_outer_loop/200) == 0:
            test_accuracy = evaluate(test_data_loader, student)
            row_dict = {
                'epoch': iter_,
                'generator_loss': g_loss,
                'student_loss': s_loss,
                'test_acc': test_accuracy
            }
            logger.log(**row_dict)
            print('Test Accuracy: ', test_accuracy)

        if (iter_ + 1) % Config.save_models_at == 0:
                generator_name = '%s_generator_itr_%d.h5' % (model_config, iter_)
                generator_filepath = os.path.join(save_dir, generator_name)
                student_name = '%s_student_itr_%d.h5' % (model_config, iter_)
                student_filepath = os.path.join(save_dir, student_name)
                generator.save_weights(generator_filepath)
                student.save_weights(student_filepath)

        stu_loss_met.reset_states()
        g_loss_met.reset_states()

def evaluate(data_loader, model, output_activations=True):
    total = 0
    correct = 0
    for inputs, labels in tqdm(data_loader):
        if output_activations:
            out, *_ = model(inputs, training=False)
        else:
            out = model(inputs, training=False)

        prob = tf.math.softmax(out, axis=-1)
        # prob = prob.numpy()

        pred = tf.argmax(prob, axis=-1)
        equality = tf.equal(pred, tf.reshape(labels, [-1]))
        correct += tf.reduce_sum(tf.cast(equality, tf.float32))
        total += equality.shape[0]

    ret = correct / tf.cast(total, tf.float32)
    return ret.numpy()

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--tdepth', type=int, required=True)
    parser.add_argument('-tw', '--twidth', type=int, required=True)
    parser.add_argument('-sd', '--sdepth', type=int, required=True)
    parser.add_argument('-sw', '--swidth', type=int, required=True)
    parser.add_argument('-tpath','--teacherpath', type=str, required=True)
    parser.add_argument('--savedir', type=str, default='zeroshot')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=10)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    zeroshot_train(args.tdepth, args.twidth, args.teacherpath, args.sdepth, args.swidth, args.seed, savedir=args.savedir)
