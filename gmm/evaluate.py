from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys


def comm_func_eval(samples, ground_truth):
    
    def ex():
        f0 = np.mean(samples, axis=0)
        f1 = np.mean(ground_truth, axis=0)
        return np.mean((f0-f1)**2)

    def exsqr():
        f0 = np.mean(samples**2, axis=0)
        f1 = np.mean(ground_truth**2, axis=0)
        return np.mean((f0-f1)**2)

    fetch = {'ex':ex(), 'exsqr':exsqr()}
    return fetch

