from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

from util import log
import os
import glob
import time

from model_bayesnn import Model
from load_data import load_uci_dataset

class Trainer(object):

    def optimize_sgd(self, loss, train_vars=None, lr=1e-2):
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        if train_vars is None:
            train_op = optimizer.minimize(loss, global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
        else:
            train_op = optimizer.minimize(loss,var_list=train_vars, 
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
        return train_op


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


    def __init__(self, config, dataset, session):
        self.config = config
        self.session = session
        self.dataset = dataset
        self.filepath = '%s-%.1f' % (
            config.method,
            config.alpha,
        )

        self.train_dir = '/data/dilin/bayesnn/train_dir/%s' % self.filepath
        #self.fig_dir = './figures/%s' % self.filepath

        #for folder in [self.train_dir, self.fig_dir]:
        for folder in [self.train_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            # clean train folder
            if self.config.clean:
                files = glob.glob(folder + '/*')
                for f in files: os.remove(f)

        #log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)

        # --- create model ---
        self.model = Model(config)

        # --- optimizer ---
        #self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.global_step = tf.Variable(0, name="global_step")

        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
        )

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.checkpoint_secs = 300  # 5 min

        self.kl_weight = tf.get_variable('kl_weight', initializer=tf.constant(1.))
        self.svgd_weight = tf.get_variable('svgd_weight', initializer=tf.constant(1.))

        self.train_op = self.optimize_adam( self.model.kl_loss, lr=self.learning_rate)
        #self.train_op = self.optimize_adagrad( self.model.kl_loss, lr=self.learning_rate)

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")

    
    def train(self):
        log.infov("Training Starts!")
        output_save_step = 1000
        buffer_save_step = 100

        self.session.run(self.global_step.assign(0)) # reset global step

        test_set = {
            'X':self.dataset.x_test,
            'y':self.dataset.y_test,
        }
        n_updates = 0
        for ep in xrange(self.config.n_epoches):
            x_train, y_train = shuffle(self.dataset.x_train, self.dataset.y_train)
            max_batches = self.config.n_train // self.config.batch_size 
            for bi in xrange(max_batches + 1):
                start = bi * self.config.batch_size
                end = min((bi+1) * self.config.batch_size, self.config.n_train)

                batch_chunk = {
                    'X': x_train[start:end],
                    'y': y_train[start:end]
                }
                step, summary, kl_loss, rmse, ll, step_time = \
                        self.run_single_step(batch_chunk)

                self.summary_writer.add_summary(summary, global_step=step)
                if n_updates % 50 == 0:
                    self.log_step_message(step, rmse, ll, step_time)
                n_updates+= 1

        test_rmse, test_ll = self.session.run(self.model.get_error_and_ll(self.model.X, self.model.y, location=self.dataset.mean_y_train, scale=self.dataset.std_y_train), feed_dict=self.model.get_feed_dict(test_set))

        write_time = time.strftime("%m-%d-%H:%M:%S")
        with open(self.config.savepath + self.config.dataset + "_test_ll_%s.txt" % (self.filepath), 'a') as f:
            f.write(repr(self.config.trial) + ',' + write_time + ',' + repr(test_ll) + '\n')

        with open(self.config.savepath + self.config.dataset + "_test_error_%s.txt" % (self.filepath), 'a') as f:
            f.write(repr(self.config.trial) + ',' + write_time + ',' + repr(test_rmse) + '\n')

        if self.config.save:
            # save model at the end
            self.saver.save(self.session,
                os.path.join(self.train_dir, 'model'),
                global_step=step)


    def run_single_step(self, batch_chunk):
        _start_time = time.time()

        fetch = [self.global_step, self.summary_op,
                 self.model.kl_loss, self.model.rmse, self.model.ll, self.train_op]

        fetch_values = self.session.run(
            fetch, feed_dict = self.model.get_feed_dict(batch_chunk)
        )

        [step, summary, kl_loss, rmse, ll] = fetch_values[:5]

        _end_time = time.time()

        return step, summary, kl_loss, rmse, ll, (_end_time - _start_time)


    def log_step_message(self, step, rmse, ll, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                #"loss: {loss:.4f} " +
                "rmse: {rmse:.4f} " +
                "ll: {ll:.4f} " +
                "({sec_per_batch:.3f} sec/batch)"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step, rmse=rmse, ll=ll,
                         sec_per_batch=step_time,
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoches', type=int, default=500, required=False)
    parser.add_argument('--method', type=str, default='alpha', choices=['alpha', 'cdf', 'negcdf'], required=True)
    parser.add_argument('--dataset', type=str, default='boston', required=True)
    parser.add_argument('--alpha', type=float, default=0, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--sample_size', type=int, default=100, required=False)
    parser.add_argument('--n_hidden', type=int, default=50, required=False)
    parser.add_argument('--trial', type=int, default=1, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3, required=False)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--savepath', type=str, default='results/', required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=1)
    config = parser.parse_args()
    
    if not config.save:
        log.warning("nothing will be saved.")

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        # intra_op_parallelism_threads=1,
        # inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(allow_growth=True),
        #device_count={'GPU': 1},
    )
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/gpu:%d'% config.gpu):
            from collections import namedtuple
            dataStruct = namedtuple("dataStruct", "x_train x_test y_train, y_test, mean_y_train, std_y_train")

            x_train, x_test, y_train, y_test, mean_y_train, std_y_train = load_uci_dataset(config.dataset, config.trial)
            config.n_train, config.dim = x_train.shape
            dataset = dataStruct(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, mean_y_train=mean_y_train, std_y_train=std_y_train)
            trainer = Trainer(config, dataset, sess)
            trainer.train()

if __name__ == '__main__':
    main()

