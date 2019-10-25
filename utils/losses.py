import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.utils import normalize

def kd_loss(p_true, p_pred):
    """
    Kullback Leibler divergence loss

    Args:
        p_true:
        p_pred:

    Ref:
    https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/losses/KLDivergence
    """
    return KLDivergence()(p_true, p_pred)

def generator_loss_fn(t_logits, s_logits, temp=1):

	loss = kd_loss(
            tf.math.softmax(t_logits / temp) ,
            tf.math.softmax(s_logits / temp))

	g_loss = -loss

	return g_loss


def student_loss_fn(t_logits, t_acts, s_logits, s_acts, beta, temp=1):
    """
    The student loss function used in
        - zero-shot learning
        - Knowledge-distillaton with attention term (KD-AT)
        - few-shot learning, few samples as KD-AT

    See Section 3.2, Eq. 1

    Args:
        t_logits: Teacher logits
        t_acts:  list of teacher activation layer output
        s_logits: Student logits
        s_acts: list of student activation layer output
        beta: hyper-parameter for tuning the weight of the attention term

    Return:
        loss
    """
    loss = kd_loss(
            tf.math.softmax(t_logits / temp) ,
            tf.math.softmax(s_logits / temp))

    if beta != 0.0:
        for t_act, s_act in zip(t_acts, s_acts):
            loss += beta*attention_loss(t_act, s_act)

    return loss

def __spatial_attention_map(act_tensor, p=2):
    """
    Spatial attention mapping function to map the activation tensor with shape
    (H, W, C) to (H, W).

    We employed:
        sum of absolute values raised to the power of 2

    The f(A_{l}) is the paper of replication

    Args:
        act_tensor: activation tensor with shape (H, W, C)
    Return:
        a spatial attention map with shape (H, W)

    Migration:
        corr. to f_act
    """

    out = tf.pow(act_tensor, p)
    out = tf.reduce_mean(out, axis=-1)
    # flatten it
    out = tf.reshape(out, [out.shape[0], -1])

    # renormalize them
    out = tf.linalg.l2_normalize(out)
    return out

def attention_loss(act1, act2):
    """
    Return the activation loss. The loss is the L2 distances between two
    activation map

    Args:
        act_map_1:
        act_map_2:

    Return:
        a floating point number representing the loss. As we use tensorflow,
        the floating point number would be a number hold in tf.Tensor

    TODO:
        check their implementation and code consistency

    Mirgration:
        to attention_loss

    Ref:
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L22
    """
    # get the activation map first
    act_map_1 = __spatial_attention_map(act1)
    act_map_2 = __spatial_attention_map(act2)

    # Calculate the L2-norm of differenes
    ret = tf.norm(act_map_2 - act_map_1, axis=-1)
    return ret


