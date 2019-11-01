import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.utils import normalize

def kldiv_loss_fn(p_true, p_pred):
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

	loss = kldiv_loss_fn(
            tf.math.softmax(t_logits / temp) ,
            tf.math.softmax(s_logits / temp))

	g_loss = -loss

	return g_loss

def cross_entropy(logits, onehot_label):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    ret = cce(onehot_label, logits)
    return ret

def knowledge_distil_loss_fn(**kwargs):
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    onehot_label = kwargs['onehot_label']
    t_logits = kwargs['t_logits']
    t_acts = kwargs['t_acts']
    s_logits = kwargs['s_logits']
    s_acts = kwargs['s_acts']
    temp = kwargs.get('temp', 1)
    # ====================================================================

    # 1. cross entopy loss
    if alpha < 1.0:
        ce_loss = cross_entropy(s_logits, onehot_label)
    else:
        ce_loss = 0

    # 2. KL Divergence
    if alpha > 0.0:
        kl_div_loss = kldiv_loss_fn(
                tf.math.softmax(t_logits / temp) ,
                tf.math.softmax(s_logits / temp))
    else:
        kl_div_loss = 0.0

    # attention loss
    att_loss = 0.0
    if beta != 0.0:
        for t_act, s_act in zip(t_acts, s_acts):
            att_loss += attention_loss(t_act, s_act)

    ret = (1 - alpha) * ce_loss + (2*alpha*temp*temp) * kl_div_loss + beta* att_loss
    return ret

def student_loss_fn(t_logits, t_acts, s_logits, s_acts, beta, temp=1):
    """
    The student loss function used in
        - zero-shot learning

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
    loss = kldiv_loss_fn(
            tf.math.softmax(t_logits / temp) ,
            tf.math.softmax(s_logits / temp))

    if beta != 0.0:
        att_loss = 0.0
        for t_act, s_act in zip(t_acts, s_acts):
            att_loss += attention_loss(t_act, s_act)

        loss += beta * att_loss

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

    Bug:
        1. use this with beta = 250 will blow up the err and the grad will explode
        2. their implementation not using beta = 250
    Ref:
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L22
    """
    # get the activation map first
    act_map_1 = __spatial_attention_map(act1)
    act_map_2 = __spatial_attention_map(act2)

    if False: # paper impl.
        # calculate vector norm of vectorized matrix
        out = tf.pow(act_map_1 - act_map_2, 2)
        out = tf.reduce_sum(out)
        ret = tf.sqrt(out)
    else:
        # their code impl
        out = tf.pow(act_map_1 - act_map_2, 2)
        out = tf.reduce_mean(out)
        ret = out
    return ret
