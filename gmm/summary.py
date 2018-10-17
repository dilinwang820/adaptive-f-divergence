import numpy as np
import os
import sys

scale = 5
dim = 2

for seed in range(100, 120):

    #for alpha in [-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0]:

    method = 'adapted'
    for alpha in [-1.0, -0.5, 0.5, 1.0]:

        prefix = '%s_%.2f' % (method, alpha)
        filepath = '%s-dim_%d' % ( prefix, dim)

        res_dir = './train_dir/seed_%d/scale_%d/%s' % (seed, scale, filepath)
        print res_dir
        with open(os.path.join(res_dir, 'results.log'), 'r') as f:
            print f.readlines()[0]

            sys.exit(0)


