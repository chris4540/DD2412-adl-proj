"""
    DONE:
    1. Impleementation of Algorithm 2
    2. Brief code optimization
    3. Name variables and functions fully
    4. Line by line programming to improve reviewers' experience

    TO DO:
    1. To improve algorithm with parallel processing:
        - Paper pseudo code is on individual image processing
        - Authors have mentioned this in their repo as an improvement but did not implement
          (https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/utils/transition_curves.py)
"""



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model as load
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow import convert_to_tensor as to_tensor
from tqdm import tqdm
import os
import sys
import time



def get_student_teacher_models(student_number=0, teacher_number=2):
    
    directory = './trained_models'
    
    models = [
        'cifar10_WRN-16-1_model.112-0.8718.h5',
        'cifar10_WRN-16-2_model.050-0.8929.h5',
        'cifar10_WRN-40-1_model.042-0.8781.h5',
        'cifar10_WRN-40-2_model.076-0.9173.h5'
    ]

    for model in models:
        assert os.path.exists('{}/{}'.format(directory, model))
    
    student = load('{}/{}'.format(directory, models[student_number]))
    teacher = load('{}/{}'.format(directory, models[teacher_number]))
    
    return student, teacher



def get_cifar10_test_data():

    (_, _), (X_test, _) = cifar10.load_data()

    X_test = X_test.astype('float32') / 255.0

    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0))
    
    return X_test



def match_beliefs(X_test, student, teacher):

    # Total steps
    K = 30
    # Total classes
    C = 10
    # Other classes
    C_other = 9
    # Pair
    P = 2

    X_test = to_tensor(X_test)

    student_preds = tf.argmax(student(X_test),-1)
    teacher_preds = tf.argmax(teacher(X_test),-1)

    common_preds = tf.reshape(tf.where(student_preds == teacher_preds), [-1])

    num_common_preds = tf.size(common_preds)

    transition_result = np.empty((num_common_preds,C_other,K,P))

    for row, i in enumerate(tqdm(common_preds)):

        X = X_test[i:i+1]

        student_pred = np.argmax(student.predict(X)[0])

        other_classes = set(digit for digit in range(C))
        other_classes.remove(student_pred)

        for col, other_class in enumerate(other_classes):

            X_step = X

            for step in range(K):
                
                with tf.GradientTape() as gradientTape:

                    gradientTape.watch(X_step)

                    student_pred = student(X_step)
                    teacher_pred = teacher(X_step)
                    
                    loss = categorical_crossentropy(tf.one_hot([other_class],C), student_pred)

                    X_step -= gradientTape.gradient(loss, X_step)
                    
                    transition_step = [student_pred[0,other_class], teacher_pred[0,other_class]]

                    transition_result[row,col,step] = transition_step
    
    return transition_result          



def compute_mean_transition_error(transition_result):

    delta = abs(transition_result[:,:,:,0] - transition_result[:,:,:,1])

    mean_delta = np.mean(delta)

    time_now = int(time.time())

    with open('./transition_results/mean_transition_error_{}.txt'.format(time_now), 'w+') as f:
        f.write('{}\n'.format(mean_delta))



def plot_transition_curves(transition_result):
    
    plt.title('Transition Curves')

    plt.xlabel('k')
    plt.ylabel('Pj')

    plt.grid(True)

    mean_transition_delta = np.mean(transition_result,(0,1))

    transition_curve = plt.plot(mean_transition_delta)

    plt.legend(transition_curve, ['Student','Teacher'])

    time_now = int(time.time())

    plt.savefig('./transition_results/transition_curve_{}.png'.format(time_now))





if __name__ == '__main__':

    n_args = len(sys.argv)

    if n_args == 1:
        student, teacher = get_student_teacher_models()
    elif n_args == 3:
        student, teacher = get_student_teacher_models(int(sys.argv[1]), int(sys.argv[2]))
    else:
        exit(1)
    
    student.trainable = False
    teacher.trainable = False

    X_test = get_cifar10_test_data()

    transition_result = match_beliefs(X_test, student, teacher)

    directory = './transition_results'
    os.makedirs(directory, exist_ok=True)
    
    compute_mean_transition_error(transition_result)

    plot_transition_curves(transition_result)

    exit(0)
