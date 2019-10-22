import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLD
from tensorflow.keras.utils import normalize

def kd_loss(teacher_logits, student_logits):
    return KLD(teacher_logits, student_logits)

def f_act(activations):
    temp1 = np.mean(np.power(activations,2), axis=-1)
    temp2 = temp1.reshape((temp1.shape[0],-1))
    return normalize(temp2)

# def attention_loss(teacher_activations, student_activations):
#     ta = f_act(teacher_activations)
#     sa = f_act(student_activations)
#     return np.mean(np.power(ta-sa,2))

def generator_loss(teacher_logits, student_logits):
    return kd_loss(teacher_logits, student_logits)

def student_loss(teacher_logits, teacher_activations, student_logits, student_activations, attn_beta):
    kld_loss = kd_loss(teacher_logits, student_logits)

    attn_loss = 0
    for i in range(len(teacher_activations)):
        attn_loss += attention_loss(teacher_activations[i], student_activations[i])

    total_loss = kld_loss + attn_beta*attn_loss
    return total_loss

# =================================================================================
# New version of loss for better readibility
def spatial_attention_map(act_tensor, p=2):
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
    act_map_1 = spatial_attention_map(act1)
    act_map_2 = spatial_attention_map(act2)

    # This is the author written in the paper
    # ret = tf.norm(act_map_2 - act_map_1, axis=-1)

    # This is the implementatin they have
    out = tf.pow(act_map_1 - act_map_2, 2)
    ret = tf.reduce_mean(out, axis=-1)
    return ret


