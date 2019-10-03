import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
from super_simple.hierarchial_model import BHSM


def generate_sample(N_GALS, seed=None):
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

    galaxies = np.array(galaxies)[
        np.random.choice(np.arange(len(galaxies)), size=N_GALS, replace=False)
    ]
    return galaxies


if __name__ == '__main__':
    loc = os.path.abspath(os.path.dirname(__file__))

    n_draws = 1000
    n_tune = 500
    n_gals = 10**2

    # generate a sample using the helper function
    galaxies = generate_sample(n_gals, seed=0)

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
        draws=n_draws,
        tune=n_tune,
        # backend='saved_gzb_bhsm_trace'
    )

    divergent = trace['diverging']

    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)

    print('Trace Summary:')
    print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))

    # save EVERYTHING
    with open('pickled_result.pickle', "wb") as buff:
        pickle.dump(
            {
                'model': bhsm, 'trace': trace,
                'n_samples': n_draws, 'n_burn': n_tune
            },
            buff
        )
