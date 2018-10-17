from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

from log_util import log
import os
import glob
import time

from evaluate import comm_func_eval

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib.backends.backend_pdf import PdfPages

from model_f_divergence import Model
from models import GaussianMixture


class Trainer(object):

    def optimize_adam(self, loss, train_vars=None, lr=1e-2):
        optimizer = tf.train.AdamOptimizer(lr)
        if train_vars is None:
            train_op = optimizer.minimize(loss, global_step=self.global_step,
                    gate_gradients=optimizer.GATE_NONE)
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars,
                    global_step=self.global_step,
                    gate_gradients=optimizer.GATE_NONE)

        return train_op


    def optimize_adagrad(self, loss, train_vars=None, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9)  #adagrad with momentum
        if train_vars is None:
            train_op = optimizer.minimize(loss, global_step=self.global_step,
                    gate_gradients=optimizer.GATE_NONE)
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars,
                    global_step=self.global_step,
                    gate_gradients=optimizer.GATE_NONE)
        return train_op


    def __init__(self, config, session):
        self.config = config
        self.session = session

        prefix = '%s_%.2f' % (config.method, config.alpha)

        self.filepath = '%s-dim_%d' % (
            prefix,
            config.dim,
        )

        self.train_dir = './train_dir/seed_%d/scale_%d/%s' % (self.config.seed, self.config.scale, self.filepath)
        self.fig_dir = './figures/seed_%d/scale_%d/%s' % (self.config.seed, self.config.scale, self.filepath)

        for folder in [self.train_dir, self.fig_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            # clean train folder
            if self.config.clean:
                files = glob.glob(folder + '/*')
                for f in files: os.remove(f)

        log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)

        # --- create model ---
        self.p_target = config.p_target
        self.model = Model( config, self.p_target)

        # --- optimizer ---
        self.global_step = tf.Variable(0, name="global_step")

        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.1,
                staircase=True,
                name='decaying_learning_rate'
        )

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.checkpoint_secs = 300  # 5 min


        self.train_op = self.optimize_adagrad(self.model.loss, 
                            train_vars=self.model.q_train_vars, lr=self.learning_rate)

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def sample_step(self, sample_size=256):
        ground_truth = self.session.run(self.p_target.sample(2000))
        samples = self.session.run(self.model.q_approx.sample(sample_size))

        return ground_truth, samples


    def evaluate_step(self, step):
        ground_truth, samples = self.sample_step()
        fetch_dict = comm_func_eval(samples, ground_truth)
        ex, exsqr = fetch_dict['ex'], fetch_dict['exsqr']
        log.infov(("[step {step:4d}] " +
                   "ex: {ex:.5f} " +
                   "exsqr: {exsqr:.5f} " ).format(
                         step=step,
                         ex=ex,
                         exsqr=exsqr,))


    def eval_run(self,):
        mean_p, var_p = self.session.run( self.p_target.model_mean_and_variance())
        mean_q, var_q = self.session.run( self.model.q_approx.model_mean_and_variance() )
        var_diff = np.mean((var_p - var_q)**2)
        var_ratio = np.mean(var_q / var_p)
        mean_diff = np.mean( (mean_p - mean_q)**2 )

        q_mu, p_mu = self.session.run([self.model.q_approx._mu, self.p_target._mu])
        from scipy.spatial import distance
        dist = distance.cdist(p_mu, q_mu, 'euclidean')
        diff = np.min(dist, axis=1).mean()

        with open(os.path.join(self.train_dir, 'results.log'), 'w') as f:
            f.write('method' + ',' + self.config.method + ',' + 'alpha' + ',' + repr(self.config.alpha) + ',' + 'scale' + ',' + repr(self.config.scale) + ',' + 'seed' + ',' + repr(self.config.seed) + ',' + 'mode_shift' + ',' + repr(diff) + ','  + 'mean' + ',' + repr(mean_diff) + ',' + 'var_diff' + ',' + repr(var_diff) + ',' + 'var_ratio' + ',' + repr(var_ratio) + '\n')


    def train(self):
        log.infov("Training Starts!")
        output_save_step = 1000

        # training q
        for n_updates in range(1, 1+self.config.max_steps):
            step, summary, loss, step_time = \
                self.run_single_step()

            self.summary_writer.add_summary(summary, global_step=step)

            if n_updates % 100 == 0: self.evaluate_step(n_updates)
            if n_updates == 1 or n_updates % output_save_step == 0:
                if self.config.save:
                    save_path = self.saver.save(self.session,
                                                os.path.join(self.train_dir, 'model'),
                                                global_step=step)
                    # scatter
                    ground_truth, samples = self.sample_step(1000)

                    ax = sns.kdeplot(ground_truth[:, 0], ground_truth[:, 1], cmap='Reds')
                    plt.scatter(samples[:, 0], samples[:, 1], color='b', alpha=0.5, s=10)
                    plt.savefig('%s/step-%d.png' % (self.fig_dir, n_updates))
                    plt.close()

        # save model at the end
        self.saver.save(self.session,
                os.path.join(self.train_dir, 'model'),
                global_step=step)

        self.eval_run()

        
    def run_single_step(self):
        _start_time = time.time()

        fetch = [self.global_step, self.summary_op, self.model.loss, self.train_op]
        fetch_values = self.session.run( fetch )
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)



from load import _simulate_mixture_target

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=10000, required=False)
    parser.add_argument('--method', type=str, default='adapted', choices=['alpha', 'adapted'], required=True)
    parser.add_argument('--alpha', type=float, default=-1, required=True)
    parser.add_argument('--sample_size', type=int, default=256, required=False)
    parser.add_argument('--dim', type=int, default=2, required=True)
    parser.add_argument('--scale', type=int, default=5, required=False)
    parser.add_argument('--proposer', type=str, default='mixture', choices=['mixture'], required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-4, required=True)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123, required=False)
    config = parser.parse_args()

    if not config.save:
        log.warning("nothing will be saved.")

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/cpu:0'):

            config.p_target = _simulate_mixture_target(dim = config.dim, seed=config.seed, val=1.0*config.scale)

            tf.set_random_seed(config.seed)
            trainer = Trainer(config, sess)
            trainer.train()

if __name__ == '__main__':
    main()

