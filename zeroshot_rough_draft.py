import tensorflow as tf
from net.generator import *
from net.wide_resnet import *
from utils.losses import *

z_dim = 100
batch_size = 128
ng_batches = 1
ns_batches = 10
attn_beta = 250
total_n_pseudo_batches = 80000
n_generator_items = ng_batches + ns_batches
total_batches = 0
student_lr = 2e-3
generator_lr = 1e-3

teacher_model = WideResidualNetwork(40, 2, input_shape=(32, 32, 3), dropout_rate=0.0)
teacher_model.load_weights('cifar10_WRN-40-2_model_92.h5')

student_model = WideResidualNetwork(16, 1, input_shape=(32, 32, 3), dropout_rate=0.0)
student_optimizer=Adam(learning_rate=student_lr)
student_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=student_lr, eta_min=0)
"""
compile student model with loss and lr_scheduler
"""
generator_model = generator(100)
generator_optimizer=Adam(learning_rate=generator_lr)
generator_scheduler = CosineAnnealingScheduler(T_max=number_of_batches, eta_max=generator_lr, eta_min=0)
"""
compile generator model with loss and lr_scheduler
"""

for total_batches in range(total_n_pseudo_batches):
    z = tf.random.normal([batch_size, z_dim])
    pseudo_images = get_gen_images(z)
    teacher_logits, *teacher_activations = get_model_outputs(teacher_model, pseudo_images, mode=0)
    
    #generator training
    for ng in range(ng_batches):
        student_logits, *student_activations = get_model_outputs(student_model, pseudo_images, mode=1)
        generator_loss = generator_loss(teacher_logits, student_logits)
        
        #################################
        # BACK PROP AND tick schedulers #
        #################################  
        
    for ns in range(ns_batches):
        student_logits, *student_activations = get_model_outputs(student_model, pseudo_images, mode=1)
        student_loss = student_loss(teacher_logits, teacher_activations, 
                                    student_logits, student_activations, attn_beta)
        
        #################################
        # BACK PROP AND tick schedulers #
        #################################   
    
    ######################################################
    ### Val accuracy computation and best model saving ###
    ######################################################    
    