import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from super_simple.hierarchial_model import BHSM, CotUniformBHSM


cot = lambda phi: 1 / np.tan(np.radians(phi))
acot = lambda a: np.degrees(np.arctan(1 / a))


def generate_sample(N_GALS=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # sample extraction
    galaxies_df = pd.read_pickle('lib/spiral_arms.pickle')
    # keep only galaxies with one arm or more
    galaxies_df = galaxies_df[galaxies_df.notna().any(axis=1)]
    # We want to scale r to have unit variance
    # get all the radial points and calculate their std
    normalization = np.concatenate(
        galaxies_df.drop('pipeline', axis=1)
            .T.unstack().dropna().apply(lambda a: a.R).values
    ).std()
    galaxies = pd.Series([
        [
            np.array((arm.t * arm.chirality, arm.R / normalization))
            for arm in galaxy.dropna()
        ]
        for _, galaxy in galaxies_df.drop('pipeline', axis=1).iterrows()
    ], index=galaxies_df.index)

    # restrict to galaxies with pitch angles between cot(4) and cot(1)
    gal_pas = galaxies_df.apply(
        lambda row: row['pipeline'].get_pitch_angle(row.dropna().values[1:])[0],
        axis=1
    ).reindex_like(galaxies)

    cot_mask = (gal_pas > cot(4)) & (gal_pas < cot(1))
    galaxies = galaxies[cot_mask]

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
                        default='cot_uniform_comparison.pickle',
                        help='Where to save output dump')

    args = parser.parse_args()

    # generate a sample using the helper function
    galaxies = generate_sample(args.ngals, seed=0)

    # initialize the models using the custom BHSM class
    bhsm = BHSM(galaxies.values)

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
