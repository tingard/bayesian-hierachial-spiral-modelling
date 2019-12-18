import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
import argparse
from super_simple.hierarchial_model import HierarchialNormalBHSM, CotUniformBHSM
import generate_sample


cot = lambda phi: 1 / np.tan(np.radians(phi))
acot = lambda a: np.degrees(np.arctan(1 / a))


LOWER_COT_BOUND = 1.19
UPPER_COT_BOUND = 4.75

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

# use the generate_sample function for consistency with other methods. Restrict
# to galaxies with pitch angle between acot(4) = 14 and acot(1) = 45
galaxies = generate_sample.generate_sample(
    n_gals=args.ngals,
    seed=0,
    pa_filter=lambda a: (a > acot(UPPER_COT_BOUND)) and (a < acot(LOWER_COT_BOUND))
)

if args.output == '':
    args.output = 'cot_uniform_n{}d{}t{}.pickle'.format(
        args.ngals or len(galaxies),
        args.ndraws,
        args.ntune,
    )


# initialize the models using the custom BHSM class
bhsm = HierarchialNormalBHSM(galaxies.values)

trace = bhsm.do_inference(
    draws=args.ndraws,
    tune=args.ntune,
    # backend='saved_gzb_bhsm_trace'
)

cot_uniform_bhsm = CotUniformBHSM(galaxies.values)

cot_uniform_trace = cot_uniform_bhsm.do_inference(
    draws=args.ndraws,
    tune=args.ntune,
    # backend='saved_gzb_bhsm_trace'
)

loo = pm.compare({bhsm.model: trace, cot_uniform_bhsm.model: cot_uniform_trace}, ic='LOO')

print('\n', loo)

# save EVERYTHING
with open(args.output, "wb") as buff:
    pickle.dump(
        {
            'normal_model': bhsm, 'normal_trace': trace,
            'cot_model': cot_uniform_bhsm, 'cot_trace': cot_uniform_trace,
            'loo': loo,
            'n_samples': args.ndraws, 'n_burn': args.ntune
        },
        buff
    )
