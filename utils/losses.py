import numpy as np
from tensorflow.keras.losses import KLD
from tensorflow.keras.utils import normalize

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

def generator_loss(teacher_logits, student_logits):
	return kd_loss(teacher_logits, student_logits)

def student_loss(teacher_logits, teacher_activations, student_logits, student_activations, attn_beta):
	kd_loss = kd_loss(teacher_logits, student_logits)

	attn_loss = 0
	for i in range(len(teacher_activations)):
		attn_loss += attention_loss(teacher_activations, student_activations)

	total_loss = kd_loss + attn_beta*attn_loss
	return total_loss




