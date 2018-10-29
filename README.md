#Variational Inference with Tail-adaptive f-Divergence


```python
def get_tail_adaptive_weights(self, n_samples, l_p, l_q, beta=-1.):
    """returns the tail-adaptive weights
    Args:
        n_samples: number of Monte Carlo samples.
        l_p: log p(x), 1-d tensor, log probability of p
        l_q: log q(x), 1-d tensor, log probability of q
        beta: magnitude, default -1
    Returns:
        Tail-adaptive weights
    """
    diff = l_p - l_q
    diff -= tf.reduce_max(diff)
    dx = tf.exp(diff)
    prob = tf.sign(tf.expand_dims(dx, 1) - tf.expand_dims(dx, 0))
    prob = tf.cast(tf.greater(prob, 0.5), tf.float32)
    wx = tf.reduce_sum(prob, axis=1) / n_samples
    wx = (1. - wx) ** beta # beta = -1; or beta = -0.5
    
    wx /= tf.reduce_sum(wx)  # self-normalization
    return tf.stop_gradient(wx)
