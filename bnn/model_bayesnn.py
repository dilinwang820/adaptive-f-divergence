from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from math import pi

class proposal_q():

    def __init__(self, config, scope_name='proposal'):
        self.config = config
        with tf.variable_scope(scope_name) as scope:
            self.param_size = (self.config.dim + 1) *  self.config.n_hidden + self.config.n_hidden + 1

            self._mu = tf.get_variable('mean', shape=(self.param_size), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=.02))

            self._log_variance = tf.get_variable('log_variance', 
                initializer=tf.constant(-10.+np.zeros((self.param_size)).astype('float32')), dtype=tf.float32)

            self._log_v_noise = tf.get_variable('log_v_noise', 
                initializer=tf.constant(np.log(1.0,).astype('float32')),
                dtype=tf.float32)

        self.params = self.get_parameters_q() 


    def draw_samples(self, n_samples):
        # (d+1) x nh + nh + 1
        ret = tf.random_normal([int(n_samples), self.param_size]) * tf.sqrt(self.params['v']) + self.params['m']
        return ret

    def get_parameters_q(self, v_prior=1., scale=1.):
        #v = tf.exp(self._log_variance)
        v = 1.0 / (scale * tf.exp(-self._log_variance ) + 1./v_prior)
        m = self._mu
        #m = scale * self._mu * tf.exp(- self._log_variance ) * v
        return {'m': m, 'v': v}

    def log_prob(self, samples, stop_grad=False):
        qv = self.params['v']
        qm = self.params['m']
        if stop_grad:
            qv = tf.stop_gradient(qv)
            qm = tf.stop_gradient(qm)

        lq = -0.5*tf.log(2*pi*qv) - 0.5*(samples - qm)**2 / qv
        return tf.reduce_sum(lq, axis=1)


class Model():

    def __init__(self, config,
                    scope_name = 'variational', is_train=True):

        self.config = config
        self.debug = {}

        self.N = self.config.n_train
        self.v_prior = 1.

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None],
        )

        self.q_approx = proposal_q(self.config)

        self.kl_loss = self.get_klqp_loss(self.config.sample_size, self.X, self.y)
        tf.summary.scalar("kl_loss", self.kl_loss)

        self.rmse, self.ll = self.get_error_and_ll(self.X, self.y, 0., 1.)
        tf.summary.scalar("batch_rmse", self.rmse)
        tf.summary.scalar("batch_ll", self.ll)


    def get_feed_dict(self, batch_chunk):
        fd = {
            self.X: batch_chunk['X'],  
            self.y: batch_chunk['y'],  
        }
        return fd

    #k : number of samples
    def predict(self, samples_q, X):
        # X:  n x d
        n, d = X.get_shape()[0].value, self.config.dim
        k = self.config.sample_size
        nh = self.config.n_hidden

        # first layer
        w1 = samples_q[:, :d * nh]  # w1: k x (nh x d)
        w1 = tf.reshape(w1, (k*nh, d))  # w1 (K x nh) x d
        b1 = samples_q[:, d*nh: (d+1)*nh]  # K x nh
        b1 = tf.reshape(b1, (1, k*nh))  # 1 x (K x nh)

        a = tf.matmul(X, w1, transpose_b=True) + b1 # n x (k * nh)
        h = tf.nn.relu(a)  # RELU, n x (k x nh)

        # second layer
        samples_q = samples_q[:, (d+1)*nh:]
        w2 = samples_q[:, :nh]  # w2: k x nh
        w2 = tf.reshape(w2, (1, k*nh)) # w2: 1 x (kxnh)
        b2 = tf.reshape(samples_q[:, nh:], (1,-1)) # b2: [k]
        out = tf.reshape( tf.reduce_sum(tf.reshape(h*w2, (-1, nh)), axis=1) , (-1, k)) + b2
        return out


    def get_error_and_ll(self, X, y, location, scale, v_prior=1.):
        v_noise = tf.exp(self.q_approx._log_v_noise) * scale**2
        samples_q = self.q_approx.draw_samples( self.config.sample_size)
        py = self.predict(samples_q, X) * scale + location
        log_factor = -0.5 * tf.log(2 * pi * v_noise) - 0.5 * (tf.expand_dims(y, 1) - py)**2 / v_noise
        ll = tf.reduce_mean(tf.reduce_logsumexp(log_factor - tf.log(1.*self.config.sample_size), axis=1))
        error = tf.sqrt(tf.reduce_mean((y - tf.reduce_mean(py, 1))**2))
        return error, ll


    def phi(self, n_samples, lpx, lqx, method, alpha=0):

        diff = lpx - lqx
        if method == 'adapted':
            # \#(t_i < t)
            diff -= tf.reduce_max(diff)
            dx = tf.exp(diff)
            prob = tf.sign(tf.expand_dims(dx, 1) - tf.expand_dims(dx, 0))  
            #prob = tf.cast(tf.equal(prob, -1), tf.float32)
            prob = tf.cast(tf.greater(prob, 0.5), tf.float32)
            wx = tf.reduce_sum(prob, axis=1) / n_samples
            wx = (1.-wx)**alpha ## alpha= -1 or alpha = -0.5
        elif method == 'alpha':
            diff = alpha * diff
            diff -= tf.reduce_max(diff)
            wx = tf.exp(diff)
        else:
            raise NotImplementedError

        wx /= tf.reduce_sum(wx)  # normalization
        return wx


    def get_klqp_loss(self, n_samples, X, y):
        v_noise = tf.exp(self.q_approx._log_v_noise)
        samples_q  = self.q_approx.draw_samples(n_samples)

        log_factor_value = 1.0 * self.N * self.log_likelihood_factor(samples_q, v_noise, X, y)

        logp0 = self.log_prior(samples_q)
        lqx = self.q_approx.log_prob(samples_q, stop_grad=True)
        lpx = logp0 + log_factor_value 

        wx = self.phi(n_samples, lpx, lqx, self.config.method, alpha=self.config.alpha)
        wx = tf.stop_gradient(wx)

        loss = tf.reduce_sum(wx * (lqx - lpx))
        return loss


    def log_likelihood_factor(self, samples_q, v_noise, X, y):

        assert X.get_shape().ndims == 2, 'illegal inputs'
        assert y.get_shape().ndims == 1, 'illegal inputs'
        py = self.predict(samples_q, X)  # n x k
        lik = -0.5 * tf.log(2 * pi * v_noise) - 0.5 * (tf.expand_dims(y, 1) - py) ** 2 /v_noise
        return tf.reduce_mean(lik, axis=0)


    def log_prior(self, samples_q):
        log_p0 = -0.5 * tf.log(2 * pi * self.v_prior) - 0.5 * samples_q **2 / self.v_prior
        return tf.reduce_sum(log_p0, axis=1)


