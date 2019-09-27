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


print('Re-creating model')
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


# cleanup for model
del agg_results
del sid_list

with bhsm.model as model:
    trace = pm.backends.text.load('saved_gzb_bhsm_trace')

    # Save the model
    try:
        pm.model_to_graphviz(model)
        plt.savefig(
            os.path.join(loc, 'plots/model.png'),
            bbox_inches='tight'
        )
        plt.close()
    except ImportError:
        pass

print('Making plots')

# Save a traceplot
pm.traceplot(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma'),
)
plt.savefig(
    os.path.join(loc, 'plots/trace.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
)
plt.savefig(
    os.path.join(loc, 'plots/posterior.png'),
    bbox_inches='tight'
)
plt.close()

# plot all the "galaxies" used
s = int(np.ceil(np.sqrt(N_GALS)))
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(8, 8), dpi=100
)
axs = [j for i in axs_grid for j in i]
for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for arm in galaxies[i]:
            o = np.argsort(arm[0])
            plt.plot(*xy_from_r_theta(arm[1][o], arm[0][o]), '.', markersize=1)
    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/sample.png'),
    bbox_inches='tight'
)
plt.close()

# make a plot showing arm predictions
with model:
    param_predictions = pm.sample_posterior_predictive(
        trace, samples=50,
        vars=(bhsm.arm_pa, bhsm.arm_c)
    )
pred_pa = param_predictions['arm_pa']
pred_c = param_predictions['c']

gal_separate_fit_params = pd.Series([])
with tqdm([galaxy for galaxy in galaxies]) as bar:
    for i, gal in enumerate(bar):
        gal_separate_fit_params.loc[i] = [
            sg.fit_log_spiral(*arm)
            for arm in gal
        ]
arm_separate_fit_params = pd.DataFrame(
    [j for _, i in gal_separate_fit_params.items() for j in i],
    columns=('pa', 'c')
)

print(arm_separate_fit_params.describe())

f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
for j in range(sum(bhsm.gal_n_arms)):
    t = bhsm.T[bhsm.arm_idx == j]
    r = bhsm.R[bhsm.arm_idx == j]
    o = np.argsort(t)
    axs[bhsm.gal_arm_map[j]].plot(
        *xy_from_r_theta(r[o], t[o]),
        'k.',
        c='C{}'.format(j % 10),
    )
for i in range(len(param_predictions)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = bhsm.T[bhsm.arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        o = np.argsort(t)
        axs[bhsm.gal_arm_map[j]].plot(
            *xy_from_r_theta(r_pred[o], t[o]),
            c='g',
            alpha=0.5,
            linewidth=3,
        )

for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for p, arm in zip(gal_separate_fit_params.iloc[i], galaxies[i]):
            R_fit = sg.log_spiral(arm[0], p[0])*np.exp(p[1])
            o = np.argsort(arm[0])
            plt.plot(*xy_from_r_theta(R_fit[o], arm[0][o]), 'r', alpha=1)

    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/prediction_comparison.png'),
    bbox_inches='tight'
)
plt.close()

try:
    import corner
    # extract the samples
    p = ('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
    names = (
        r'\phi_\mathrm{global}}',
        r'\sigma_\mathrm{global}',
        r'\sigma_\mathrm{gal}}',
        r'\sigma_r'
    )
    postsamples = np.vstack([trace[k] for k in p]).T
    print('Number of posterior samples is {}'.format(postsamples.shape[0]))
    fig = corner.corner(postsamples, labels=['${}$'.format(k) for k in p])
    fig.savefig('plots/corner.png')
except ImportError:
    import sys
    sys.exit(1)
