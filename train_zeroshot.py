"""
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.generator import NavieGenerator
from utils.cosine_anealing import CosineAnnealingScheduler
from utils.losses import kd_loss
from utils.losses import student_loss
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
import numpy as np

# TODO: use Config class
z_dim = 100
batch_size = 128
ng_batches = 1
ns_batches = 10
attn_beta = 250
total_n_pseudo_batches = 20
n_generator_items = ng_batches + ns_batches
total_batches = 0
student_lr = 2e-3
generator_lr = 1e-3
number_of_batches = 10

teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
teacher.load_weights('saved_models/cifar10_WRN-40-2_model.h5')
teacher.trainable = False

student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
student_optimizer = Adam(learning_rate=student_lr)
# student_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=student_lr, eta_min=0)

generator = NavieGenerator(input_dim=100)
generator_optimizer = Adam(learning_rate=generator_lr)
# generator_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=generator_lr, eta_min=0)

# Generator loss metrics
g_loss_met = tf.keras.metrics.Mean()
# Student loss metrics
stu_loss_met = tf.keras.metrics.Mean()


def cosine_lr_schedule(epoch, T_max, eta_max, eta_min=0):
    lr = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return lr

for total_batches in range(total_n_pseudo_batches):
    # sample from latern space to make an image
    z = tf.random.normal([batch_size, z_dim])

    # Generator training
    generator.trainable = True
    student.trainable = False
    for ng in range(ng_batches):
        with tf.GradientTape() as tape:
            pseudo_imgs = generator(z)
            t_logits, *t_acts = teacher(pseudo_imgs)
            s_logits, *s_acts = student(pseudo_imgs)

            # calculate the generator loss
            generator_loss = kd_loss(tf.math.softmax(t_logits),
                                     tf.math.softmax(s_logits))

        # The grad for generator
        grads = tape.gradient(generator_loss, generator.trainable_weights)

        # cosine annealing for learning rate
        generator_optimizer.learning_rate = cosine_lr_schedule(total_batches, total_n_pseudo_batches, generator_lr)

        # update the generator paramter with the gradient
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        g_loss_met(generator_loss)

        if total_batches % 2 == 0:
            print('step %s: generator mean loss = %s' % (total_batches, g_loss_met.result()))
    # ==========================================================================

    # Student training
    generator.trainable = False
    student.trainable = True
    for ns in range(ns_batches):

        t_logits, *t_acts = teacher(pseudo_imgs)
        with tf.GradientTape() as tape:
            s_logits, *s_acts = student(pseudo_imgs)
            std_loss = student_loss(t_logits, t_acts, s_logits, s_acts, attn_beta)

        # The grad for student
        grads = tape.gradient(std_loss, student.trainable_weights)

        # Update learning rate
        student_optimizer.learning_rate = cosine_lr_schedule(total_batches, total_n_pseudo_batches, student_lr)
        student_optimizer.apply_gradients(zip(grads, student.trainable_weights))

        stu_loss_met(std_loss)

        if total_batches % 2 == 0:
            print('step %s: studnt mean loss = %s' % (total_batches, stu_loss_met.result()))
