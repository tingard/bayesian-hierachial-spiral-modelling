import os
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
import super_simple.sample_generation as sg
# import argparse
from tqdm import tqdm
from super_simple.hierarchial_model import BHSM


# used to limit sample for testing
N_GALS = 25

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
    trace = pm.sample(2000, tune=1000, target_accept=0.95, max_treedepth=20,
                      init='advi+adapt_diag', trace=db)

print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))
