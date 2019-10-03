import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from super_simple.hierarchial_model import BHSM


def generate_sample(N_GALS=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # sample extraction
    agg_results = pd.read_pickle('lib/aggregation_results.pickle')

    # scale r to have unit variance
    rs = np.concatenate([
        arm.R for gal in agg_results.Arms.values for arm in gal
    ])
    normalization = rs.std()
    galaxies = [
        [
            np.array((arm.t * arm.chirality, arm.R / normalization))
            for arm in galaxy
        ]
        for galaxy in agg_results.Arms.values
    ]
    if N_GALS > 0:
        galaxies = np.array(galaxies)[
            np.random.choice(
                np.arange(len(galaxies)),
                size=N_GALS, replace=False
            )
        ]
        if len(galaxies) < N_GALS:
            warnings.warn('Sample contains fewer galaxies than specified')
    return galaxies


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Run hierarchial model and save output'
        )
    )
    parser.add_argument('--ngals', '-N', default=-1, type=int,
                        help='Number of galaxies in sample')
    parser.add_argument('--ntune', default=500, type=int,
                        help='Number of tuning steps to take')
    parser.add_argument('--ndraws', default=1000, type=int,
                        help='Number of posterior draws to take')
    parser.add_argument('--output', '-o', metavar='/path/to/file.pickle',
                        default='',
                        help='Where to save output dump')

    args = parser.parse_args()
    if args.output == '':
        args.output = 'n{}d{}t{}.pickle'.format(
            args.ngals,
            args.ndraws,
            args.ntune,
        )
    # generate a sample using the helper function
    galaxies = generate_sample(args.ngals, seed=0)

    # initialize the model using the custom BHSM class
    bhsm = BHSM(galaxies)
    print(bhsm.data.describe())

    # save the model
    try:
        with bhsm.model as model:
            pm.model_to_graphviz(bhsm.model).render(
                'plots/model', view=False
            )
    except ImportError:
        pass

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
