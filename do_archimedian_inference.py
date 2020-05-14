import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from super_simple.hierarchial_model import ArchimedianBHSM
import generate_sample


import argparse

parser = argparse.ArgumentParser(
    description=(
        'Run hierarchial model and save output'
    )
)
parser.add_argument('--ngals', '-N', default=None, type=int,
                    help='Number of galaxies in sample')
parser.add_argument('--ntune', default=2000, type=int,
                    help='Number of tuning steps to take')
parser.add_argument('--ndraws', default=20000, type=int,
                    help='Number of posterior draws to take')
parser.add_argument('--output', '-o', metavar='/path/to/file.pickle',
                    default='',
                    help='Where to save output dump')

args = parser.parse_args()

# generate a sample using the helper function
galaxies = generate_sample.generate_sample(n_gals=args.ngals, seed=0)

if args.output == '':
    args.output = 'archimedian_n{}d{}t{}.pickle'.format(
        args.ngals or len(galaxies),
        args.ndraws,
        args.ntune,
    )

# initialize the model using the custom BHSM class
bhsm = ArchimedianBHSM(galaxies)

trace = bhsm.do_inference(
    draws=args.ndraws,
    tune=args.ntune,
)

try:
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)
except KeyError:
    pass

print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='r_hat', ascending=False))

# save EVERYTHING
with open(args.output, "wb") as buff:
    pickle.dump(
        {
            'model': bhsm, 'trace': trace,
            'n_samples': args.ndraws, 'n_burn': args.ntune
        },
        buff
    )
