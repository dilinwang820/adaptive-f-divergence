import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim', required=True, help='dim')
parser.add_argument('--log', required=True, help='filename')
opt = parser.parse_args()


if not opt.log.endswith('log'):
    print ('Please select a .log file')
    sys.exit(0)

filename = opt.log.replace('.log','')
results = {}
with open('%s.log' % filename) as fp:
    for line in fp:
        values = line.strip().split(' ')

        row = {}
        for i in range(0, len(values), 2):
            row[values[i]] = values[i+1]
        
        method = row['method']
        alpha = row['alpha']
        scale = int(row['scale'])

        key = method + alpha
        seed = int(row['seed'])
        mode_diff = float(row['mode_diff'])
        var_diff = float(row['var_diff'])
        ex = float(row['ex'])

        if key not in results:
            results[key] = {}
        if scale not in results[key]: 
            results[key][scale] = {}
            results[key][scale]['mode'] = []
            results[key][scale]['var'] = []
            results[key][scale]['ex'] = []

        results[key][scale]['mode'].append( mode_diff )
        results[key][scale]['var'].append( var_diff )
        results[key][scale]['ex'].append( ex )


keys = []
#for alpha in ['-500.0', '-2.0', '-1.5', '-1.0', '0.5', '0.0', '0.5', '1.0', '1.5', '2.0', '500.0']:
#for alpha in ['-1.0', '-0.5', '0.5', '1.0']:
for alpha in ['-1.0']:
    keys.append('negcdf%s' % alpha)

#for alpha in ['-2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0']:
for alpha in ['-1.0', '0.0', '0.5', '1.0', '2.0']:
#for alpha in ['-1.0', '0.0', '0.5', '1.0']:
    keys.append('alpha%s' % alpha)

for key in keys:
    for scale in [0, 1 , 2, 3, 4, 5]:
    #for scale in [5]:
        m_vals = results[key][scale]['mode']
        v_vals = np.log10(results[key][scale]['var'])
        ex = np.log10(results[key][scale]['ex'])
        print (key, opt.dim, scale, \
                    np.mean(m_vals), np.std(m_vals)/np.sqrt(len(m_vals)), \
                    np.mean(v_vals), np.std(v_vals)/np.sqrt(len(v_vals)), \
                    np.mean(ex), np.std(ex) / np.sqrt(len(ex)))


