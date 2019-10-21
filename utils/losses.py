import numpy as np
from tensorflow.keras.losses import KLD
from tensorflow.keras.utils import normalize
from models import *

def kd_loss(teacher_logits, student_logits):
	return KLD(teacher_logits, student_logits)

def f_act(activations):
	temp1 = np.mean(np.power(activations,2), axis=-1)
	temp2 = temp1.reshape((x.shape[0],-1))
	return normalize(temp2)

def attention_loss(teacher_activations, student_activations):
	ta = f_act(teacher_activations)
	sa = f_act(student_activations)
	return np.mean(np.power(ta-sa,2))



