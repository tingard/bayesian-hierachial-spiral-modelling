import os
import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
import sample_generation as sg
import argparse
from tqdm import tqdm


loc = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--ngals', '-n', metavar='N', default=25,
                    type=int, help='Number of galaxies in sample')
parser.add_argument('--mu', metavar='N', default=20,
                    type=str, help='Global mean pitch angle')
parser.add_argument('--sd', metavar='N', default=5,
                    type=str, help='Inter-galaxy pitch angle std')
parser.add_argument('--sd2', metavar='N', default=10,
                    type=str, help='Intra-galaxy pitch angle std')

args = parser.parse_args()

# Base parameter definition
N_GALS = args.ngals
BASE_PA = args.mu
INTER_GAL_SD = args.sd
INTRA_GAL_SD = args.sd2
N_POINTS = 100
PA_LIMS = (0.1, 60)

print((
    'Making sample of {} galaxies with global mean pa {:.2f}'
    '\nInter-galaxy pitch angle std: {:.2e}'
    '\nIntra-galaxy pitch angle std: {:.2e}'
).format(N_GALS, BASE_PA, INTER_GAL_SD, INTRA_GAL_SD))

gal_pas = st.truncnorm.rvs(*PA_LIMS, loc=BASE_PA, scale=INTER_GAL_SD,
                           size=N_GALS)
gal_n_arms = [np.random.poisson(0.75) + 2 for i in range(N_GALS)]

print('Input galaxies:')
print(pd.DataFrame({
    'Pitch angle': gal_pas,
    'Arm number': gal_n_arms
}).describe())

# map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is from the
# 3rd galaxy)
gal_arm_map = np.concatenate([
    np.tile(i, n) for i, n in enumerate(gal_n_arms)
])

# Generate our galaxies
galaxies = [
    sg.gen_galaxy(gal_n_arms[i], gal_pas[i], INTRA_GAL_SD, N=N_POINTS)
    for i in range(N_GALS)
]

# Create an array containing needed information in a stacked form
point_data = np.concatenate([
    np.stack((
        arm_T,
        arm_R,
        np.tile(sum(gal_n_arms[:gal_n]) + arm_n, len(arm_T)),
        np.tile(gal_n, len(arm_T))
    ), axis=-1)
    for gal_n, galaxy in enumerate(galaxies)
    for arm_n, (arm_T, arm_R) in enumerate(galaxy)
])

T, R, arm_idx, gal_idx = point_data.T
arm_idx = arm_idx.astype(int)

# ensure the arm indexing makes sense
assert np.all((np.unique(arm_idx) - np.arange(sum(gal_n_arms))) == 0)

# Define Stochastic variables
with pm.Model() as model:
    # model r = a * exp(tan(phi) * t) + sigma as r = exp(b * t + c) + sigma,
    # and have a uniform prior on phi rather than the gradient!
    # Note our test values should not be the optimum, as then there is little
    # Gradient information to work with! It's about finding a balance between
    # the initial log probability and sufficient gradient for NUTS

    # Global mean pitch angle
    global_pa_mu = pm.Uniform(
        'pa',
        lower=0, upper=90,
        testval=20
    )

    # inter-galaxy dispersion
    global_pa_sd = pm.InverseGamma('pa_sd', alpha=1, beta=3)

    # intra-galaxy dispersion
    gal_pa_sd = pm.InverseGamma('gal_pa_sd', alpha=1, beta=3)

    # arm offset parameter
    arm_c = pm.Cauchy('c', alpha=0, beta=10, shape=len(gal_arm_map),
                      testval=np.tile(0, len(gal_arm_map)))

    # radial noise (degenerate witxh error on pitch angle I think...)
    sigma = pm.HalfCauchy('sigma', beta=0.2, testval=0.1)

# Define Dependent variables
with model:
    # we want this:
    # gal_pa_mu = pm.TruncatedNormal(
    #     'gal_pa_mu',
    #     mu=global_pa_mu, sd=global_pa_sd,
    #     lower=0.1, upper=60,
    #     shape=n_gals,
    # )
    # arm_pa = pm.TruncatedNormal(
    #     'arm_pa_mu',
    #     mu=gal_pa_mu[gal_arm_map], sd=gal_pa_sd[gal_arm_map],
    #     lower=0.1, upper=60,
    #     shape=len(gal_arm_map),
    # )
    # Specified in a non-centred way:

    gal_pa_mu_offset = pm.Normal(
        'gal_pa_mu_offset',
        mu=0, sd=1, shape=N_GALS,
        testval=np.tile(0, N_GALS)
    )
    gal_pa_mu = pm.Deterministic(
        'gal_pa_mu',
        global_pa_mu + gal_pa_mu_offset * global_pa_sd
    )

    # use a Potential for the truncation, pm.Potential('foo', N) simply adds N
    # to the log likelihood
    pm.Potential(
        'gal_pa_mu_bound',
        (
            tt.switch(tt.all(gal_pa_mu > 0), 0, -np.inf)
            + tt.switch(tt.all(gal_pa_mu < 90), 0, -np.inf)
        )
    )

    arm_pa_mu_offset = pm.Normal(
        'arm_pa_mu_offset',
        mu=0, sd=1, shape=sum(gal_n_arms),
        testval=np.tile(0, sum(gal_n_arms))
    )
    arm_pa = pm.Deterministic(
        'arm_pa',
        gal_pa_mu[gal_arm_map] + arm_pa_mu_offset * gal_pa_sd
    )
    pm.Potential(
        'arm_pa_mu_bound',
        (
            tt.switch(tt.all(arm_pa > 0), 0, -np.inf)
            + tt.switch(tt.all(arm_pa < 90), 0, -np.inf)
        )
    )

    # convert to a gradient for a linear fit
    arm_b = tt.tan(np.pi / 180 * arm_pa)
    arm_r = tt.exp(arm_b[arm_idx] * T + arm_c[arm_idx])
    pm.Potential(
        'arm_r_bound',
        tt.switch(tt.all(arm_r < 1E4), 0, -np.inf)
    )
    # likelihood function
    likelihood = pm.Normal(
        'Likelihood',
        mu=arm_r,
        sigma=sigma,
        observed=R
    )

# it's important we now check the model specification, namely do we have any
# problems with logp being undefined?
with model:
    print(model.check_test_point())

# Sampling
with model:
    trace = pm.sample(2000, tune=1000, target_accept=0.95, max_treedepth=20,
                      init='advi+adapt_diag')

    # Save the model
    try:
        pm.model_to_graphviz(model)
        plt.savefig(
            os.path.join(loc, 'plots/many_galaxies_model.png'),
            bbox_inches='tight'
        )
        plt.close()
    except ImportError:
        pass

trace_saved = pm.save_trace(trace)

# Save a traceplot
pm.traceplot(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma'),
    lines=(
        ('pa_scaled', {}, BASE_PA),
        ('pa_sd', {}, INTER_GAL_SD),
        ('gal_pa_sd', {}, INTRA_GAL_SD),
    )
)
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_trace.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
)
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_posterior.png'),
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
            plt.plot(*xy_from_r_theta(arm[1], arm[0]))
    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies.png'),
    bbox_inches='tight'
)
plt.close()

# make a plot showing arm predictions
with model:
    param_predictions = pm.sample_posterior_predictive(
        trace, samples=50,
        vars=(arm_pa, arm_c)
    )
pred_pa = param_predictions['arm_pa']
pred_c = param_predictions['c']

f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
for j in range(sum(gal_n_arms)):
    t = T[arm_idx == j]
    r = R[arm_idx == j]
    axs[gal_arm_map[j]].plot(
        *xy_from_r_theta(r, t),
        'k.',
        c='C{}'.format(j % 10),
    )
for i in range(len(param_predictions)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = T[arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        axs[gal_arm_map[j]].plot(
            *xy_from_r_theta(r_pred, t),
            c='k',
            alpha=0.5,
            linewidth=1,
        )
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_predictions.png'),
    bbox_inches='tight'
)
plt.close()

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

f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
for j in range(sum(gal_n_arms)):
    t = T[arm_idx == j]
    r = R[arm_idx == j]
    axs[gal_arm_map[j]].plot(
        *xy_from_r_theta(r, t),
        'k.',
        c='C{}'.format(j % 10),
    )
for i in range(len(param_predictions)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = T[arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        axs[gal_arm_map[j]].plot(
            *xy_from_r_theta(r_pred, t),
            c='g',
            alpha=0.5,
            linewidth=3,
        )

for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for p, arm in zip(gal_separate_fit_params.iloc[i], galaxies[i]):
            R_fit = sg.log_spiral(arm[0], p[0])*np.exp(p[1])
            plt.plot(*xy_from_r_theta(R_fit, arm[0]), 'r', alpha=1)

    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_prediction_comparison.png'),
    bbox_inches='tight'
)
plt.close()

print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))
