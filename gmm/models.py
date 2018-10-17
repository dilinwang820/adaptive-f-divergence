from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distribution_util

import sys
import numpy as np
from math import pi


'''
Original implementation of Gumble-softmax
https://github.com/ericjang/gumbel-softmax
'''
def sample_gumbel(n_samples, c, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform([n_samples, c], minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, n_samples, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(n_samples, logits.get_shape()[0].value)
    return tf.nn.softmax( y / temperature, 1)

def gumbel_softmax(logits, n_samples, temperature=0.1, hard=False):
    assert logits.get_shape().ndims == 1, 'illegal inputs'
    """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, int(n_samples), temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


class MultiVariateGaussian():

    def __init__(self, mu, log_var):
        self._mu = mu
        self._var = tf.exp(log_var)

        self._A = 1. / self._var
        self._lnD = tf.reduce_sum(log_var)

        self.dim = self._mu.get_shape()[0].value
        assert len(list(self._mu.get_shape())) == 1, 'illegal inputs'

    def log_gradient(self, x):   # A, inverse of covariance matrix
        return -(x - self._mu) *  self._A

    def log_prob(self, x, stop_grad=False):
        if stop_grad:
            mu = tf.stop_gradient(self._mu)
            A = tf.stop_gradient(self._A)
        else:
            mu = self._mu
            A = self._A
        assert len(list(x.get_shape())) == 2, 'illegal inputs'
        x = x - mu
        sum_sq_dist_times_inv_cov = tf.reduce_sum((x * A) * x, axis=1)
        return -0.5 * sum_sq_dist_times_inv_cov - self.logZ(stop_grad)


    def logZ(self, stop_grad=False):
        if stop_grad:
            lnD = tf.stop_gradient(self._lnD)
        else:
            lnD = self._lnD
        ln2piD = tf.log(2 * np.pi) * self.dim
        #log_coefficients = ln2piD + tf.log(self._D) 
        log_coefficients = ln2piD + lnD
        return 0.5 * log_coefficients


    def sample(self, n_samples,with_init_noise=False):
        raw = tf.random_normal([int(n_samples), self.dim])
        ret = self._mu + raw * tf.sqrt(self._var)
        ret.set_shape((int(n_samples), self.dim))
        if with_init_noise:
            return ret, raw
        return ret
    


class GaussianMixture():

    def __init__(self, n_components, mu, log_var, weights=None, is_train=True):
        assert mu.get_shape().ndims == 2 and log_var.get_shape().ndims == 2, 'illegal inputs'
        self.is_train = is_train
        self.n_components = n_components

        self._mu = mu
        self.dim = self._mu.get_shape()[1].value

        self._log_var = log_var
        self._var = tf.exp(self._log_var)

        if self.is_train:
            assert weights is not None, 'illegal inputs'
            self._weights = tf.nn.softmax(weights)
            self._logits = tf.log(self._weights)
            self._cat = lambda x: gumbel_softmax(self._logits, x) # x: n_samples
        else:
            if weights is None:
                weights = tf.ones(shape=(self.n_components,), dtype=tf.float32)
            self._weights = weights / tf.reduce_sum(weights)
            self._cat = tf.distributions.Categorical(probs=self._weights)


    def _sum_log_exp(self, X, weights, mu, log_var):

        diff = tf.expand_dims(X, 0) - tf.expand_dims(mu, 1)  # c x n x d
        diff_times_inv_cov = diff * tf.expand_dims(1./ tf.exp(log_var), 1)  # c x n x d
        sum_sq_dist_times_inv_cov = tf.reduce_sum(diff_times_inv_cov * diff, axis=2)  # c x n 
        ln2piD = tf.log(2 * np.pi) * self.dim

        lnD = tf.reduce_sum(log_var, axis=1)
        log_coefficients = tf.expand_dims(ln2piD + lnD, 1) # c x 1
        log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
        log_weighted = log_components + tf.expand_dims(tf.log(weights), 1)  # c x n + c x 1
        log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
        return log_weighted, log_shift


    def log_gradient(self, X):  

        # X: n_samples x d; mu: c x d; cov: c x d x d
        x_shape = X.get_shape()
        assert len(list(x_shape)) == 2, 'illegal inputs'
    
        def posterior(X):
            log_weighted, log_shift = self._sum_log_exp(X, self._weights, self._mu, self._log_var)
            prob = tf.exp(log_weighted - log_shift) # c x n
            prob = prob / tf.reduce_sum(prob, axis=0, keep_dims=True)
            return prob
    
        diff = tf.expand_dims(X, 0) - tf.expand_dims(self._mu, 1)  # c x n x d
        diff_times_inv_cov = -diff * tf.expand_dims(1./self._var, 1)  # c x n x d
    
        P = posterior(X)  # c x n
        score = tf.matmul(
            tf.expand_dims(tf.transpose(P, perm=[1, 0]), 1), # n x 1 x c
            tf.transpose(diff_times_inv_cov, perm=[1, 0, 2]) # n x c x d
        ) 
        return tf.squeeze(score, axis=1)


    def log_prob(self, X, stop_grad=False):  
        # X: n_samples x d; 
        x_shape = X.get_shape()
        assert len(list(x_shape)) == 2, 'illegal inputs'

        if stop_grad:
            weights = tf.stop_gradient(self._weights)
            mu = tf.stop_gradient(self._mu)
            log_var = tf.stop_gradient(self._log_var)
        else:
            weights = self._weights
            mu = self._mu
            log_var = self._log_var

        log_weighted, log_shift = self._sum_log_exp(X, weights, mu, log_var)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, axis=0, keep_dims=True)
        logp = tf.log(exp_log_shifted_sum) + log_shift # 1 x n
        return tf.squeeze(logp)


    def sample(self, n_samples):
        n_samples = int(n_samples)

        if self.is_train:
            cat_probs = self._cat(n_samples)  # n x c
            agg_mu = tf.reduce_sum(tf.expand_dims(cat_probs, 2) * self._mu, axis=1) # n x d 
            agg_var = tf.reduce_sum(tf.expand_dims(cat_probs, 2) * self._var, axis=1) # n x d

            raw = tf.random_normal([n_samples, self.dim])
            ret = agg_mu + tf.sqrt(agg_var) * raw # n x d 

            #samples_class = [None for _ in range(self.n_components)]
            #for c in range(self.n_components):
            #    raw = tf.random_normal([n_samples, self.dim])
            #    samples_class_c = self._mu[c] + raw * tf.sqrt(self._sigma[c]) #tf.matmul(raw, tf.transpose(self._scale[c]))
            #    samples_class[c] = samples_class_c
            #samples_class = tf.stack(samples_class) # c x n x d
            #ret = tf.reduce_sum(tf.expand_dims(cat_probs, 2) * tf.transpose(samples_class, [1,0,2]), axis=1)
        else:
            cat_samples = self._cat.sample(n_samples) # n x 1

            samples_raw_indices = array_ops.reshape(
                      math_ops.range(0, n_samples), cat_samples.get_shape().as_list())

            partitioned_samples_indices = data_flow_ops.dynamic_partition(
                      data=samples_raw_indices,
                      partitions=cat_samples,
                      num_partitions=self.n_components)

            samples_class = [None for _ in range(self.n_components)]
            for c in range(self.n_components):
                n_class = array_ops.size(partitioned_samples_indices[c])
                raw = tf.random_normal([n_class, self.dim])
                samples_class_c = self._mu[c] + raw * tf.sqrt(self._var[c])
                samples_class[c] = samples_class_c

            # Stitch back together the samples across the components.
            ret = data_flow_ops.dynamic_stitch(
                            indices=partitioned_samples_indices, data=samples_class)
            ret.set_shape((int(n_samples), self.dim))
        return ret


    def model_mean_and_variance(self):
        mu = tf.reduce_sum( tf.expand_dims(self._weights, 1) * self._mu, axis=0 )
        var = 0
        for i in range(self.n_components):
            var += ( self._weights[i] * ((self._mu[i] - mu)**2 + self._var[i]) )
        return mu, var


def main(_):

    with tf.Graph().as_default(), tf.Session() as sess:

        n_components = 2
        dim = 2

        #X0 = tf.constant(np.asarray([[1, 2], [1, 3], [1,4]]), dtype=tf.float32)
        #mu0 = tf.constant(np.asarray([[1, 0], [0, 2]]), dtype=tf.float32)
        #log_sigma = np.asarray([[.2, .1], [.1, .3]]).astype('float32')
        #weights0 = tf.constant(np.asarray([.2, .8]), dtype=tf.float32)

        #model = GaussianMixture(n_components, mu0, log_sigma, weights0, is_train=True)

        #print(sess.run(model._sigma))
        #print(sess.run(model._mu))
        #print(sess.run(model._weights))

        X0 = tf.constant(np.asarray([[1, 2], [1, 3], [1,4]]), dtype=tf.float32)
        mu0 = tf.constant(np.asarray([2, 3]), dtype=tf.float32)
        log_sigma = np.asarray([0, 0]).astype('float32')
        model = MultiVariateGaussian(mu0, log_sigma)

        print(sess.run(model._mu))
        print(sess.run(model._sigma))
        grads = sess.run( model.log_gradient(X0))
        print (grads)
        print (sess.run(model.log_prob(X0)))

        #with tf.variable_scope('random') as scope:
        #    random_offset = tf.get_variable('random_offset', dtype=tf.float32,
        #      initializer=tf.constant(np.random.uniform(0., 2, (10,)).astype('float32')),
        #    )
        #    weights = tf.get_variable('weights', dtype=tf.float32,
        #        initializer=tf.constant(np.random.normal(0., 1., (10, 20)).astype('float32')),
        #    )

        #t = tf.get_variable('t', dtype=tf.float32,
        #        initializer=tf.constant(np.ones((4, 2)).astype('float32')),
        #)

        #variables_names = [v.name for v in tf.trainable_variables()]
        ##print (sess.run(variables_names))
        #print (variables_names)

        #train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                    "random/")

        #ft = t**2
        #out = tf.stack([tf.gradients(ft[:,i], [t])[0] for i in range(2)])
        #print (sess.run(out))
        #    
        ##logp = sess.run(model.logp_gmm(X0) )
        ##print (logp)
        ##grad = sess.run(model.score_gmm(X0) )
        ##print (grad)

        #X0 = tf.constant(np.asarray([[1, 2], [1, 3], [1,4]]), dtype=tf.float32)
        #mu0 = tf.constant(np.asarray([1, 2]), dtype=tf.float32)
        #cov0 = tf.constant(np.asarray([[2,1],[1, 2]]), dtype=tf.float32)

        #scale0 = tf.cholesky(cov0)
        #print( sess.run(tf.matmul(scale0, tf.transpose(scale0))) ) 
        #logp = sess.run(logp_mvn(X0, mu0, scale0) )
        #print (logp)
        #grad = sess.run(score_mvn(X0, mu0, scale0) )
        #print (grad)

        #samples = sess.run(samples_mvn(3000, mu0, scale0))
        #print (np.mean(samples, 0))
        #print (np.cov(samples.T))

        #a = tf.constant(np.arange(1, 13, dtype=np.int32),
        #                shape=[2, 2, 3])
        #b = tf.constant(np.arange(13, 25, dtype=np.int32),
        #                shape=[2, 2, 3])
        #print(sess.run(tf.matmul(a, b, transpose_b=True)))
        #print(sess.run(tf.matmul(a, tf.transpose(b, perm=[0,2,1]))))
        #dist = tf.distributions.Categorical(probs=weights0)
        #print( sess.run(dist.sample(int(10))) )

if __name__ == "__main__":
    tf.app.run()
