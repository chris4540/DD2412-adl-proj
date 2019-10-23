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
class Config:
    """
    This config should be static as we follow the paper
    """
    # the random seed dimension
    z_dim = 100
    batch_size = 128
    # The number of inner loop to train generator to have an adversarial example
    n_g_in_loop = 1
    # The number of inner loop to align teacher and student with kd-at
    n_s_in_loop = 10
    # The weigting of the attention term
    beta = 250
    # The number of steps of the outer loop. The "N" in Algorithm 1
    n_outer_loop = 20

    # init learing rates
    student_init_lr = 2e-3
    generator_init_lr = 1e-3

## Teacher
teacher = WideResidualNetwork(40, 2, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
teacher.load_weights('saved_models/cifar10_WRN-40-2_model.h5')
teacher.trainable = False

## Student
student = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
student_optimizer = Adam(learning_rate=CosineDecay(
                            Config.student_init_lr,
                            decay_steps=Config.n_outer_loop*Config.n_g_in_loop))
## Generator
generator = NavieGenerator(input_dim=100)
## TODO: double check the annuealing setting
generator_optimizer = Adam(learning_rate=CosineDecay(
                            Config.generator_init_lr,
                            decay_steps=Config.n_outer_loop*Config.n_s_in_loop))

# Generator loss metrics
g_loss_met = tf.keras.metrics.Mean()
# Student loss metrics
stu_loss_met = tf.keras.metrics.Mean()


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
            loss = kd_loss(tf.math.softmax(t_logits),
                           tf.math.softmax(s_logits))

        # The grad for generator
        grads = tape.gradient(loss, generator.trainable_weights)

        # update the generator paramter with the gradient
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        g_loss_met(loss)

        if iter_ % 2 == 0:
            print('step %s: generator mean loss = %s' % (iter_, g_loss_met.result()))
    # ==========================================================================

    # Student training
    generator.trainable = False
    student.trainable = True
    for ns in range(Config.n_s_in_loop):

        t_logits, *t_acts = teacher(pseudo_imgs)
        with tf.GradientTape() as tape:
            s_logits, *s_acts = student(pseudo_imgs)
            loss = student_loss_fn(t_logits, t_acts, s_logits, s_acts, Config.beta)

        # The grad for student
        grads = tape.gradient(loss, student.trainable_weights)

        # Apply grad for student
        student_optimizer.apply_gradients(zip(grads, student.trainable_weights))

        stu_loss_met(loss)

        if iter_ % 2 == 0:
            print('step %s: studnt mean loss = %s' % (iter_, stu_loss_met.result()))
