import math
import time
import numpy as np

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
import pprint

base_dir = '/home/dilin/Dropbox/code/tf_matching/renyi/VRbound/BayesianNN/data/'

def load_uci_dataset(dataset, i):
    # We load the data
    datapath = base_dir + dataset + '/'
    data = np.loadtxt(datapath + 'data.txt')
    index_features = np.loadtxt(datapath + 'index_features.txt').astype('int')
    index_target = np.loadtxt(datapath + 'index_target.txt').astype('int')

    X = data[ : , index_features.tolist() ]
    y = data[ : , index_target.tolist() ]

    # We load the indexes of the training and test sets
    index_train = np.loadtxt(datapath + "index_train_{}.txt".format(i)).astype('int')
    index_test = np.loadtxt(datapath + "index_test_{}.txt".format(i)).astype('int')
    # load training and test data
    X_train = X[ index_train.tolist(), ]
    y_train = y[ index_train.tolist() ]
    X_test = X[ index_test.tolist(), ]
    y_test = y[ index_test.tolist() ]

    # We normalize the features
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train

    y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
    y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))

    return X_train, X_test, np.squeeze(y_train), np.squeeze(y_test), mean_y_train, std_y_train


def main(_):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='alpha', choices=['alpha', 'cdf', 'negcdf', 'log'], required=False)
    parser.add_argument('--alpha', type=float, default=0, required=False)
    parser.add_argument('--beta', type=float, default=-1, required=False)
    parser.add_argument('--dim', type=int, default=13, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--sample_size', type=int, default=100, required=False)
    parser.add_argument('--n_hidden', type=int, default=100, required=False)
    parser.add_argument('--k', type=int, default=10, required=False)
    config = parser.parse_args()

    from model_bayesnn import Model
    x_train, x_test, y_train, y_test, mean_y_train, std_y_train = load_uci_dataset('boston', 1)
    config.n_train = len(x_train)
    config.dim = x_train.shape[1]

    with tf.Graph().as_default(), tf.Session() as sess:

        model = Model(config)

        tf.global_variables_initializer().run()
        batch_chunk = {
            'X': x_train[:config.batch_size],
            'y': y_train[:config.batch_size]
        }
        print(
            sess.run(model.kl_loss, feed_dict=model.get_feed_dict(batch_chunk))
        )
        batch_test = {
            'X': x_test,
            'y': y_test
        }
 
        print(
            sess.run(model.get_error_and_ll(model.X, model.y, location=mean_y_train, scale=std_y_train), feed_dict=model.get_feed_dict(batch_test)) 
        )


if __name__ == "__main__":
    tf.app.run()



