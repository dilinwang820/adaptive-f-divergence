from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

from util import log
import os
import time
from evaluate import comm_func_eval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from model_f_divergence import Model
from models import GaussianMixture

class Evaler(object):

    def __init__(self, config):
        self.config = config
        self.p_target = config.p_target

        self.model = Model( config, self.p_target)

        self.global_step = tf.Variable(0, name="global_step")
        self.step_op = tf.no_op(name='step_no_op')

        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True),
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=3)

        if self.config.checkpoint is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            return 
        self.checkpoint_path = tf.train.latest_checkpoint(config.checkpoint)
        log.info("Checkpoint path : %s", self.checkpoint_path)

   
    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        var_p = self.session.run( self.p_target.model_variance())
        var_q = self.session.run( self.model.q_approx.model_variance() )
        var_diff = np.mean((var_p - var_q)**2)

        real, appx = self.session.run([self.p_target.sample(10000), self.model.q_approx.sample(2000)])
        res = comm_func_eval(appx, real)

        q_mu, p_mu = self.session.run([self.model.q_approx._mu, self.p_target._mu])
        from scipy.spatial import distance
        dist = distance.cdist(p_mu, q_mu, 'euclidean')
        diff = np.min(dist, axis=1).mean()
        print ('method', self.config.method, 'alpha', self.config.alpha, 'scale', self.config.scale, 'seed', self.config.seed,  \
                        'mode_diff', diff, 'var_diff', var_diff, 'p_var', np.mean(var_p), 'q_var', np.mean(var_q), \
                        'ex', res['ex'], 'exsqr', res['exsqr']) #, 'p_var', var_p, 'q_var', var_q)

        if self.config.plot:
            weights = self.session.run( self.model.phi(len(appx), tf.constant(appx), self.config.method, self.config.alpha))
            #np.savez('weights_method_%s_alpha_%.2f.npz' % (self.config.method, self.config.alpha), w=weights)

            np.savez('samples/method_%s_alpha_%.2f.npz' % (self.config.method, self.config.alpha), real=real, appx=appx, weights=weights)

            #ax = sns.kdeplot(real[:, 0], color='r', linewidth=4.0)
            #ax = sns.kdeplot(appx[:, 0], color='b', linewidth=4.0)
            ##ax = sns.kdeplot(real[:, 0], real[:, 1], cmap='Reds', linewidths=1.0, n_levels=12)
            ##ax = sns.kdeplot(appx[:, 0], appx[:, 1], cmap='Blues', linewidths=2.0, n_levels=12)
            ##plt.scatter(real[:1000, 0], real[:1000, 1], color='r', alpha=0.5, s=10)
            ##plt.xlim(-8, 8)
            ##plt.ylim(-8, 8)
            #pp = PdfPages('method_%s_alpha_%.2f.pdf' % (self.config.method, self.config.alpha))
            #pp.savefig()
            #pp.close()

            #f, axarr = plt.subplots()
            #ax = sns.kdeplot(real[:, 0], real[:, 1], shade=False, cmap='Reds', n_levels=20)
            #plt.xlim(-8, 8)
            #plt.ylim(-8, 8)
            #plt.axis('equal')
            #ax.axis('off')
            ##plt.savefig('true.png')
            #pp = PdfPages('true.pdf')
            #pp.savefig()
            #pp.close()
            #plt.close()

            #f, axarr = plt.subplots()
            #ax = sns.kdeplot(appx[:, 0], appx[:, 1], shade=False, cmap='Blues', n_levels=20)
            #plt.xlim(-8, 8)
            #plt.ylim(-8, 8)
            #plt.axis('equal')
            #ax.axis('off')
            ##plt.savefig('appx.png')
            #pp = PdfPages('method_%s_alpha_%.2f.pdf' % (self.config.method, self.config.alpha))
            #pp.savefig()
            #pp.close()
            #plt.close()

            ### Add labels to the plot
            ##red = sns.color_palette("Reds")[-2]
            ##blue = sns.color_palette("Blues")[-2]
            ##ax.text(3.8, 4.5, "p", size=16, color=red)
            ##ax.text(2.5, 8.2, "q", size=16, color=blue)
            ##plt.savefig('kde.png')


from load import _simulate_mixture_target

def main():

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--max_steps', type=int, default=10000, required=False)
    parser.add_argument('--method', type=str, default='alpha', choices=['alpha', 'cdf', 'negcdf', 'log', 'ess'], required=True)
    parser.add_argument('--alpha', type=float, default=0, required=True)
    parser.add_argument('--sample_size', type=int, default=256, required=False)
    parser.add_argument('--dim', type=int, default=2, required=True)
    parser.add_argument('--scale', type=int, default=5, required=False)
    parser.add_argument('--proposer', type=str, default='mixture', choices=['gaussian', 'mixture'], required=False)
    parser.add_argument('--gradient', type=str, default='rp', choices=['rp', 'sf'], required=True)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    #parser.add_argument('--learning_rate', type=float, default=5e-4, required=True)
    #parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    #parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123, required=False)
    parser.add_argument('--gpu', type=int, default=1)
    config = parser.parse_args()

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/gpu:%d'% config.gpu):

            config.p_target = _simulate_mixture_target(dim = config.dim, seed=config.seed, val=1.0*config.scale)

            tf.set_random_seed(config.seed)
            evaler = Evaler(config)
            evaler.eval_run()

if __name__ == '__main__':
    main()
