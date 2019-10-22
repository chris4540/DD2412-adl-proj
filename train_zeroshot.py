import tensorflow as tf
tf.compat.v1.enable_eager_execution(config=None, device_policy=None,execution_mode=None)
from net.generator import NavieGenerator
from utils.cosine_anealing import CosineAnnealingScheduler
from utils.losses import *
from tensorflow.keras.optimizers import Adam
from net.wide_resnet import WideResidualNetwork
# from train_scratch import *
import numpy as np

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

# teacher_model = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
# teacher_model.load_weights('saved_models/cifar10_WRN-16-1_model.005.h5')

teacher_model = WideResidualNetwork(40, 2, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
teacher_model.load_weights('saved_models/cifar10_WRN-40-2_model.h5')

student_model = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0, output_activations=True)
student_optimizer=Adam(learning_rate=student_lr)
student_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=student_lr, eta_min=0)

generator_model = NavieGenerator(input_dim=100)
generator_optimizer=Adam(learning_rate=generator_lr)
generator_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=generator_lr, eta_min=0)

student_model.trainable = True
teacher_model.trainable = False
generator_model.trainable = True

gen_loss_metric = tf.keras.metrics.Mean()
stu_loss_metric = tf.keras.metrics.Mean()

def cosine_lr_schedule(epoch, T_max, eta_max, eta_min=0):
    lr = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return lr

for total_batches in range(total_n_pseudo_batches):
    # sample from latern space to make an image
    z = tf.random.normal([batch_size, z_dim])

    # Generator training
    for ng in range(ng_batches):
        with tf.GradientTape() as gtape:
            pseudo_images = generator_model(z)
            teacher_logits, *teacher_activations = teacher_model(pseudo_images)
            student_logits, *student_activations = student_model(pseudo_images)
            generator_loss = kd_loss(tf.math.softmax(teacher_logits), tf.math.softmax(student_logits))

        gen_grads = gtape.gradient(generator_loss, generator_model.trainable_weights)

        #cosine annealing for learning rate
        generator_optimizer.learning_rate = cosine_lr_schedule(total_batches, total_n_pseudo_batches, generator_lr)

        #update gradient
        generator_optimizer.apply_gradients(zip(gen_grads, generator_model.trainable_weights))

        gen_loss_metric(generator_loss)

        if total_batches % 2 == 0:
            print('step %s: generator mean loss = %s' % (total_batches, gen_loss_metric.result()))

    # Student training
    for ns in range(ns_batches):
        with tf.GradientTape() as stape:
            pseudo_images = generator_model(z)
            teacher_logits, *teacher_activations = teacher_model(pseudo_images)
            student_logits, *student_activations = student_model(pseudo_images)
            std_loss = student_loss(teacher_logits, teacher_activations,
                                student_logits, student_activations, attn_beta)

        st_grads = stape.gradient(std_loss, student_model.trainable_weights)

        student_optimizer.learning_rate = cosine_lr_schedule(total_batches, total_n_pseudo_batches, student_lr)
        student_optimizer.apply_gradients(zip(st_grads, student_model.trainable_weights))

        stu_loss_metric(std_loss)

        if total_batches % 2 == 0:
            print('step %s: studnt mean loss = %s' % (total_batches, stu_loss_metric.result()))
