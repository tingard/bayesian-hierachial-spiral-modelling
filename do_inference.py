import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from super_simple.hierarchial_model import UniformBHSM


def generate_sample(N_GALS=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # sample extraction
    galaxies_df = pd.read_pickle('lib/spiral_arms.pickle')\
        .drop('pipeline', axis=1)
    # keep only galaxies with one arm or more
    galaxies_df = galaxies_df[galaxies_df.notna().any(axis=1)]
    # We want to scale r to have unit variance
    # get all the radial points and calculate their std
    normalization = np.concatenate(
        galaxies_df.T.unstack().dropna().apply(lambda a: a.R).values
    ).std()
    galaxies = pd.Series([
        [
            np.array((arm.t * arm.chirality, arm.R / normalization))
            for arm in galaxy.dropna()
        ]
        for _, galaxy in galaxies_df.iterrows()
    ], index=galaxies_df.index)
    if N_GALS is not None and N_GALS > 0 and N_GALS < len(galaxies):
        galaxies = galaxies.iloc[
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
    galaxies = generate_sample(args.ngals, seed=0)
    if args.output == '':
        args.output = 'n{}d{}t{}.pickle'.format(
            args.ngals or len(galaxies),
            args.ndraws,
            args.ntune,
        )

    # initialize the model using the custom BHSM class
    bhsm = BHSM(galaxies)
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
