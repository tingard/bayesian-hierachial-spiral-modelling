import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from super_simple.hierarchial_model import HierarchialNormalBHSM
import argparse
import generate_sample


parser = argparse.ArgumentParser(
    description=(
        'Run hierarchial model and save output'
    )
)
parser.add_argument('--ngals', '-N', default=None, type=int,
                    help='Number of galaxies in sample')
parser.add_argument('--ntune', default=500, type=int,
                    help='Number of tuning steps to take')
parser.add_argument('--ndraws', default=1000, type=int,
                    help='Number of posterior draws to take')
parser.add_argument('--output', '-o', metavar='/path/to/file.pickle',
                    default='',
                    help='Where to save output dump')

args = parser.parse_args()

# generate a sample using the helper function
galaxies = generate_sample.generate_sample(n_gals=args.ngals, seed=0)

if args.output == '':
    args.output = 'hierarchial_n{}d{}t{}.pickle'.format(
        args.ngals or len(galaxies),
        args.ndraws,
        args.ntune,
    )

# initialize the model using the custom BHSM class
bhsm = HierarchialNormalBHSM(galaxies)
print(bhsm.data.describe())

trace = bhsm.do_inference(
    draws=args.ndraws,
    tune=args.ntune,
    # backend='saved_gzb_bhsm_trace'
)

divergent = trace['diverging']

print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent %.1f' % divperc)

print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))

# save EVERYTHING
with open(args.output, "wb") as buff:
    pickle.dump(
        {
            'model': bhsm, 'trace': trace,
            'n_samples': args.ndraws, 'n_burn': args.ntune
        },
        buff
    )
