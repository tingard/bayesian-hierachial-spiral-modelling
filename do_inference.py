import pickle
import os
import numpy as np
import pandas as pd
import pymc3 as pm
from super_simple.hierarchial_model import BHSM


# used to limit sample for testing
N_GALS = 36

loc = os.path.abspath(os.path.dirname(__file__))

# sample extraction
agg_results = pd.read_pickle('lib/aggregation_results.pickle')
sid_list = agg_results.index.values

# scale r from 0 to 1
max_r = max(np.max(arm.R) for gal in agg_results.Arms.values for arm in gal)
galaxies = [
    [
        np.array((arm.t * arm.chirality, arm.R / max_r))
        for arm in galaxy
    ]
    for galaxy in agg_results.Arms.values
    if len(galaxy) > 1
]

# reduce the sample size for testing purposes
np.random.seed(0)
galaxies = np.array(galaxies)[
    np.random.choice(np.arange(len(galaxies)), size=N_GALS, replace=False)
]

# initialize the model using the custom BHSM class
bhsm = BHSM(galaxies)

# it's important we now check the model specification, namely do we have any
# problems with logp being undefined?
with bhsm.model as model:
    print(model.check_test_point())

# Sampling
with bhsm.model as model:
    db = pm.backends.Text('saved_gzb_bhsm_trace')
    trace = bhsm.do_inference(draws=500, tune=500, backend='saved_gzb_bhsm_trace')
    divergent = trace['diverging']

print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent %.1f' % divperc)


print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))

# save EVERYTHING
with open('pickled_result.pickle', "wb") as buff:
    pickle.dump({'model': bhsm, 'trace': trace, 'divergent': divergent}, buff)
