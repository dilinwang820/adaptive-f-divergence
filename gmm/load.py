from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from models import GaussianMixture


def _simulate_mixture_target(n_components=10, dim = 2, val=5., seed=123):
    assert dim > 1, 'illegal inputs'
    with tf.variable_scope('p_target') as scope:
        np.random.seed(seed)
        mu0 = tf.get_variable('mu', initializer=np.random.uniform(-val, val, size=(n_components, dim)).astype('float32'), dtype=tf.float32,  trainable=False)

        log_var0 = tf.zeros((n_components, dim))
        weights0 = tf.ones(n_components) / n_components
        p_target = GaussianMixture(n_components, mu0, log_var0, weights0, is_train=False)

        return p_target



