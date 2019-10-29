import tensorflow as tf

def evaluate(data_loader, model, output_activations=True):
    total = 0
    correct = 0.0
    for inputs, labels in data_loader:
        if output_activations:
            out, *_ = model(inputs, training=False)
        else:
            out = model(inputs, training=False)

        prob = tf.math.softmax(out, axis=-1)

        pred = tf.argmax(prob, axis=-1)
        equality = tf.equal(pred, tf.reshape(labels, [-1]))
        correct = correct + tf.reduce_sum(tf.cast(equality, tf.float32))
        total = total + tf.size(equality)

    total = tf.cast(total, tf.float32)
    ret = correct / total
    return ret
