import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

method = ['adapted,-1.0', 'adapted,-0.5', 'alpha,0.0', 'alpha,0.5', 'alpha,1.0']
datasets = ['boston']

#table = {}
#naval_test_error_alpha-0.0.txt
for ds in datasets:
    #if ds not in table: table[ds] = {}
    rmse, ll = [], []
    # alpha approach
    for m in method:
        #if m not in table[ds]: table[ds][m] = {}
        t, v = m.split(',')
        test_error_path = '%s_test_error_%s-%s.txt' % (ds, t, v)
        test_ll_path = '%s_test_ll_%s-%s.txt' % (ds, t, v)
        test_error = np.loadtxt(test_error_path, delimiter=',', usecols=[0,2])
        test_ll = np.loadtxt(test_ll_path, delimiter=',', usecols=[0,2])
        #table[ds][m]['rmse'] = np.mean(test_error[:,1])
        #table[ds][m]['ll'] = np.mean(test_ll[:,1])
        n_samples = len(test_error)
        rmse.append((np.mean(test_error[:,1]), np.std(test_error[:,1]) / np.sqrt(n_samples)))
        ll.append((np.mean(test_ll[:,1]), np.std(test_error[:,1])/ np.sqrt(n_samples)))

    print ds
    print method
    print ds,',', [ '%.3f#%.3f' % (v, s) for (v,s) in rmse]
    print ds,',', [ '%.3f#%.3f' % (v, s) for (v,s) in ll]
