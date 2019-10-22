"""
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.generator import NavieGenerator
from utils.cosine_anealing import CosineAnnealingScheduler
from utils.losses import kd_loss
from utils.losses import student_loss_fn
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
from tensorflow.keras.experimental import CosineDecay
import numpy as np

# TODO: use Config class
z_dim = 100
batch_size = 128
ng_batches = 1
ns_batches = 10
attn_beta = 250
total_n_pseudo_batches = 3
n_generator_items = ng_batches + ns_batches
student_lr = 2e-3
generator_lr = 1e-3
number_of_batches = 3

teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
teacher.load_weights('saved_models/cifar10_WRN-40-2_model.h5')
teacher.trainable = False

student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
student_optimizer = Adam(learning_rate=CosineDecay(student_lr, number_of_batches))

generator = NavieGenerator(input_dim=100)
generator_optimizer = Adam(learning_rate=CosineDecay(generator_lr, number_of_batches))

# Generator loss metrics
g_loss_met = tf.keras.metrics.Mean()
# Student loss metrics
stu_loss_met = tf.keras.metrics.Mean()


for total_batches in range(total_n_pseudo_batches):
    # sample from latern space to make an image
    z = tf.random.normal([batch_size, z_dim])

    # Generator training
    generator.trainable = True
    student.trainable = False
    for ng in range(ng_batches):
        with tf.GradientTape() as tape:
            pseudo_imgs = generator(z)
            t_logits, *_ = teacher(pseudo_imgs)
            s_logits, *_ = student(pseudo_imgs)

            # calculate the generator loss
            loss = kd_loss(tf.math.softmax(t_logits),
                                     tf.math.softmax(s_logits))

        # The grad for generator
        grads = tape.gradient(loss, generator.trainable_weights)

        # update the generator paramter with the gradient
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        g_loss_met(loss)

        print('step %s: generator mean loss = %s' % (total_batches, g_loss_met.result()))
    # ==========================================================================

    # Student training
    generator.trainable = False
    student.trainable = True
    for ns in range(ns_batches):

        t_logits, *t_acts = teacher(pseudo_imgs)
        with tf.GradientTape() as tape:
            s_logits, *s_acts = student(pseudo_imgs)
            loss = student_loss_fn(tf.math.softmax(t_logits), t_acts, tf.math.softmax(s_logits), s_acts, attn_beta)

        # The grad for student
        grads = tape.gradient(loss, student.trainable_weights)

        # Apply grad for student
        student_optimizer.apply_gradients(zip(grads, student.trainable_weights))

        stu_loss_met(loss)

        print('step %s-%s: studnt mean loss = %s' % (total_batches, ns, stu_loss_met.result()))
