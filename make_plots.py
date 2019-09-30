import os
import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
import super_simple.sample_generation as sg
import seaborn as sns
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


# Fit each arm separately
print('Fitting individually')
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


print('Building model')
# initialize the model using the custom BHSM class
bhsm = BHSM(galaxies)

# cleanup to save RAM
del agg_results
del sid_list

with bhsm.model as model:
    trace = pm.backends.text.load('saved_gzb_bhsm_trace')


print('Getting predictions')
with bhsm.model as model:
    param_predictions = pm.sample_posterior_predictive(
        trace, samples=100,
        vars=(bhsm.arm_pa, bhsm.arm_c, bhsm.global_pa_mu, bhsm.global_pa_sd)
    )
pred_pa = param_predictions['arm_pa']
pred_c = param_predictions['c']

pred_mu_phi = param_predictions['pa']
pred_sigma_phi = param_predictions['pa_sd']


print('Making plots')


def plot_sample(galaxies, axs, **kwargs):
    assert len(axs) >= len(galaxies)
    kwargs.setdefault('alpha', 0.7)
    kwargs.setdefault('markersize', 2)
    for i, ax in enumerate(axs):
        plt.sca(ax)
        try:
            for j, arm in enumerate(galaxies[i]):
                o = np.argsort(arm[0])
                plt.plot(
                    *xy_from_r_theta(arm[1][o], arm[0][o]),
                    '.',
                    color='C{}'.format(j % 10),
                    **kwargs
                )
        except IndexError:
            pass


var_names = ('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
names = (
    r'$\phi_\mathrm{global}}$',
    r'$\sigma_\mathrm{global}$',
    r'$\sigma_\mathrm{gal}}$',
    r'$\sigma_r$'
)


def traceplot(trace, var_names=[], names=None):
    assert (names is None) or (len(var_names) == len(names))
    f, ax = plt.subplots(nrows=len(var_names), ncols=2, dpi=100)
    for i, p in enumerate(var_names):
        plt.sca(ax[i][0])
        for j in range(4):
            chain = trace.get_values(p, burn=1000, chains=[j])
            sns.kdeplot(chain)
            ax[i][1].plot(chain, alpha=0.25)
            if names is not None:
                plt.title(names[i])
    plt.tight_layout()


print('\tPlotting traceplot')
# # this use too much RAM, so we define our own above
# pm.traceplot(
#     trace,
#     var_names=var_names
# )
traceplot(trace, var_names, names)
plt.savefig(
    os.path.join(loc, 'plots/trace.png'),
    bbox_inches='tight'
)
# plt.close()
# print('\tPlotting posterior')
# # Save a posterior plot
# pm.plot_posterior(
#     trace,
#     var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
# )
# plt.savefig(
#     os.path.join(loc, 'plots/posterior.png'),
#     bbox_inches='tight'
# )
# plt.close()

print('\tPlotting sample')
# plot all the "galaxies" used
s = int(np.ceil(np.sqrt(N_GALS)))
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
plot_sample(galaxies, axs)
plt.savefig(
    os.path.join(loc, 'plots/sample.png'),
    bbox_inches='tight'
)
plt.close()

print('\tPlotting predictions')
# make a plot showing arm predictions
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
plot_sample(galaxies, axs)

# plot the individually fit arms
for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for p, arm in zip(gal_separate_fit_params.iloc[i], galaxies[i]):
            R_fit = sg.log_spiral(arm[0], p[0])*np.exp(p[1])
            o = np.argsort(arm[0])
            xy = xy_from_r_theta(R_fit[o], arm[0][o])
            plt.plot(*xy, 'k', alpha=0.7, linewidth=2)

    except IndexError:
        pass

# plot the posterior predictions
for i in range(len(pred_pa)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = bhsm.T[bhsm.arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        o = np.argsort(t)
        xy = xy_from_r_theta(r_pred[o], t[o])
        axs[bhsm.gal_arm_map[j]].plot(
            *xy,
            c='r',
            alpha=3 / len(pred_pa),
            linewidth=1,
        )
plt.savefig(
    os.path.join(loc, 'plots/prediction_comparison.png'),
    bbox_inches='tight'
)
plt.close()

print('\tPlotting corner plot')
# plot a jointplot, if corner is available
try:
    import corner
    # extract the samples
    postsamples = np.vstack([
        trace[k][1000:]  # trace.get_values(k, burn=1000)
        for k in var_names
    ]).T
    print('\t\tNumber of posterior samples is {}'.format(postsamples.shape[0]))
    fig = corner.corner(postsamples, labels=names)
    fig.savefig('plots/corner.png')
except ImportError:
    try:
        df = pm.backends.tracetab.trace_to_dataframe(
            trace[1000:], chains=None, varnames=var_names
        )
        sns.pairplot(df)
        plt.savefig('plots/corner_seaborn.png')
        plt.close()
    except ImportError:
        import sys
        sys.exit(0)


print('\tPlotting posterior of pitch angle')
plt.figure(figsize=(12, 4), dpi=100)
x = np.linspace(0, 90, 1000)
ys = [
    st.norm.pdf(x, loc=pred_mu_phi[i], scale=pred_sigma_phi[i])
    for i in range(len(pred_mu_phi))
]
y = np.add.reduce(ys, axis=0) / len(ys)
plt.fill_between(
    x, np.zeros(len(x)), y,
    color='r', alpha=0.5
)
plt.plot(x, y, 'r')
# for i in range(len(pred_mu_phi)):
#     y = st.norm.pdf(x, loc=pred_mu_phi[i], scale=pred_sigma_phi[i])
#     plt.fill_between(
#         x, np.zeros(len(x)), y,
#         color='k', alpha=1/len(pred_mu_phi)
#     )
empirical_dist = st.norm.pdf(
    x,
    loc=arm_separate_fit_params['pa'].mean(),
    scale=arm_separate_fit_params['pa'].std(),
)
plt.plot(x, empirical_dist, 'k--')
plt.savefig('plots/pa_realizations.png', bbox_inches='tight')
plt.close()
