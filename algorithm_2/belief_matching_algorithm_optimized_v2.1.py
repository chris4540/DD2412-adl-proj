import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model as load
from tensorflow.keras.backend import categorical_crossentropy as ce
from tensorflow import convert_to_tensor as ct
from tqdm import tqdm
import os
import sys
import time



def get_student_teacher_models(s=0, t=2):
    
    dir = './trained_models'
    
    mdls = ['cifar10_WRN-16-1_model.112-0.8718.h5',
            'cifar10_WRN-16-2_model.050-0.8929.h5',
            'cifar10_WRN-40-1_model.042-0.8781.h5',
            'cifar10_WRN-40-2_model.076-0.9173.h5']

    for m in mdls:
        assert os.path.exists('{}/{}'.format(dir,m))
    
    m_s = load('{}/{}'.format(dir,mdls[s]))
    m_t = load('{}/{}'.format(dir,mdls[t]))
    
    return m_s, m_t
    


def get_cifar10_test():

    (_, _), (X_te, _) = cifar10.load_data()

    X_te = X_te.astype('float32') / 255.0

    X_te = (X_te - X_te.mean(axis=0)) / (X_te.std(axis=0))
    
    return X_te



def match_beliefs(X_te=None, m_s=None, m_t=None):

    X_te = ct(X_te)
    ix = np.where(np.argmax(m_s.predict(X_te),-1) == np.argmax(m_t.predict(X_te),-1))[0]
    T = np.empty((len(ix),9,30,2))

    for r,i in enumerate(tqdm(ix)):

        X = X_te[i:i+1]
        y_s = np.argmax(m_s.predict(X)[0])

        y_cs = set(d for d in range(10))
        y_cs.remove(y_s)

        for y,y_c in enumerate(y_cs):

            X_c = X

            for k in range(30):
                
                with tf.GradientTape() as g:

                    g.watch(X_c)

                    Y_s = m_s(X_c)
                    Y_t = m_t(X_c)
                    
                    X_c -= g.gradient(ce(tf.one_hot([y_c],10), Y_s), X_c)
                    
                    T[r,y,k] = [Y_s[0,y_c], Y_t[0,y_c]]
    
    compute_mte(T)
    
    return T          



def compute_mte(T):

    dir = './transition_curves'
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open('{}/mte_{}.txt'.format(dir, int(time.time())), 'w+') as f:
        f.write('{}\n'.format(np.mean(abs(T[:,:,:,0] - T[:,:,:,1]))))



def create_transition_curves(T):
    
    plt.title('Transition Curves')
    plt.xlabel('k')
    plt.ylabel('Pj')
    plt.grid(True)

    plt.legend(plt.plot(np.mean(T,(0,1))), ['Student','Teacher'])

    plt.savefig('./transition_curves/curve_{}.png'.format(int(time.time())))





if __name__ == '__main__':

    m_s, m_t = get_student_teacher_models() if len(sys.argv)==1 else get_student_teacher_models(int(sys.argv[1]), int(sys.argv[2]))
    
    m_s.trainable = False
    m_t.trainable = False

    create_transition_curves(match_beliefs(get_cifar10_test(), m_s, m_t))

    exit(0)
