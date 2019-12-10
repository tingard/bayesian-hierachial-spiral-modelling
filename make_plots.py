import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import matplotlib.pyplot as plt
from gzbuilder_analysis.aggregation.spirals import xy_from_r_theta
import super_simple.sample_generation as sg
import seaborn as sns
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
    description=(
        'Make plots of results from hierarchial model'
    )
)
parser.add_argument('--input', default='pickled_result.pickle',
                    help='Pickled result from running the do_inference.py script')
parser.add_argument('--output', '-o', metavar='/path/to/directory',
                    default='', help='Where to save plots')
args = parser.parse_args()

try:
    with open(args.input, 'rb') as f:
        saved = pickle.load(f)
except IOError:
    print('Invalid input file')
    sys.exit(1)

if args.output == '':
    args.output = 'plots_{}'.format(
        args.input.split('.')[0]
    ) if ('.' in args.input) else 'plots'

if not os.path.isdir(args.output):
    os.makedirs(args.output)


# extract the needed parameters from the saved state
bhsm = saved['model']
galaxies = bhsm.galaxies
trace = saved['trace']
# prefix = [i for i in trace.varnames if 'phi_gal' in i][0].replace('phi_gal', '')
prefix = bhsm.model.name
prefix += ('_' if len(prefix) > 0 else '')
print(f'Identified model name "{prefix}"')

n_draws = saved.get('n_draws', 500)
n_tune = saved.get('n_tune', 500)

print('N tune: {}, N draws: {}'.format(n_tune, n_draws))


# PREPARATION
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


print('Getting predictions')
with bhsm.model as model:
    param_predictions = pm.sample_posterior_predictive(
        trace, samples=100,
        vars=[bhsm.phi_arm, bhsm.c, bhsm.mu_phi, bhsm.sigma_phi]
    )
pred_pa = param_predictions[f'{prefix}phi_arm']
pred_c = param_predictions[f'{prefix}c']

pred_mu_phi = param_predictions[f'{prefix}mu_phi']
pred_sigma_phi = param_predictions[f'{prefix}sigma_phi']


# PLOTTING
print('Making plots')

var_names = (f'{prefix}mu_phi', f'{prefix}sigma_phi', f'{prefix}sigma_r')
names = (
    r'$\mu_\phi$',
    r'$\sigma_\phi$',
    r'$\sigma_r$',
)


def plot_sample(galaxies, axs, **kwargs):
    assert len(axs) >= len(galaxies)
    kwargs.setdefault('alpha', 0.7)
    kwargs.setdefault('markersize', 2)
    for i, ax in enumerate(axs):
        plt.sca(ax)
        try:
            for j, arm in enumerate(galaxies.iloc[i]):
                o = np.argsort(arm[0])
                plt.plot(
                    *xy_from_r_theta(arm[1][o], arm[0][o]),
                    '.',
                    color='C{}'.format(j % 10),
                    **kwargs
                )
        except IndexError:
            pass


def traceplot(trace, var_names=[], names=None):
    assert (names is None) or (len(var_names) == len(names))
    f, ax = plt.subplots(
        nrows=len(var_names),
        figsize=(12, 3*len(var_names)),
        ncols=2,
        dpi=100
    )
    for i, p in enumerate(var_names):
        plt.sca(ax[i][0])
        for j in range(trace.nchains):
            chain = trace.get_values(p, chains=[j])
            sns.kdeplot(chain)
            ax[i][1].plot(chain, alpha=0.25)
            if names is not None:
                plt.title(names[i])
    plt.tight_layout()


print('\tPlotting model')
try:
    with bhsm.model as model:
        pm.model_to_graphviz(bhsm.model).render(
            os.path.join(args.output, 'model'),
            view=False,
            format='pdf',
            cleanup=True,
        )
except ImportError:
    pass


print('\tPlotting traceplot')
# # this use too much RAM, so we define our own above
# pm.traceplot(
#     trace,
#     var_names=var_names
# )
traceplot(trace, var_names, names)
plt.savefig(
    os.path.join(args.output, 'trace.png'),
    bbox_inches='tight'
)
plt.close()

print('\tPlotting sample')
# plot all the "galaxies" used
s = int(np.ceil(np.sqrt(min(25, len(galaxies)))))
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
plot_sample(galaxies.iloc[:len(axs)], axs)
plt.savefig(
    os.path.join(args.output, 'sample.png'),
    bbox_inches='tight'
)
plt.close()

print('\tPlotting predictions')
# make a plot showing arm predictions
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=200
)
axs = [j for i in axs_grid for j in i]
plot_sample(galaxies.iloc[:len(axs)], axs, alpha=0.6)

# plot the posterior predictions
# for each sample (of 100 drawn samples)
for i in range(len(pred_pa)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    # for each arm in this sample
    for j in range(len(arm_pa)):
        if bhsm.gal_arm_map[j] >= len(axs):
            continue
        t = bhsm.data['theta'][i].values
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        o = np.argsort(t)
        xy = xy_from_r_theta(r_pred[o], t[o])
        axs[bhsm.gal_arm_map[j]].plot(
            *xy,
            c='k',
            alpha=3 / len(pred_pa),
            linewidth=0.5,
        )

# plot the individually fit arms
for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for p, arm in zip(gal_separate_fit_params.iloc[i], galaxies.iloc[i]):
            R_fit = sg.log_spiral(arm[0], p[0])*np.exp(p[1])
            o = np.argsort(arm[0])
            xy = xy_from_r_theta(R_fit[o], arm[0][o])
            plt.plot(*xy, 'r:', linewidth=1)

    except IndexError:
        pass

plt.savefig(
    os.path.join(args.output, 'prediction_comparison.png'),
    bbox_inches='tight'
)
plt.close()

print('\tPlotting corner plot')
# plot a jointplot, if corner is available
try:
    import corner
    # extract the samples
    postsamples = np.vstack([
        trace[k]  # trace.get_values(k, burn=1000)
        for k in var_names
    ]).T
    print('\t\tNumber of posterior samples is {}'.format(postsamples.shape[0]))
    fig = corner.corner(postsamples, labels=names)
    fig.savefig(os.path.join(args.output, 'corner.png'))
except ImportError:
    try:
        df = pm.backends.tracetab.trace_to_dataframe(
            trace, chains=None, varnames=var_names
        )
        sns.pairplot(df)
        plt.savefig(os.path.join(args.output, 'corner_seaborn.png'))
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
plt.plot(x, y, 'r', label='Hierarchial model')
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
plt.plot(x, empirical_dist, 'k--', label='Separately fit pitch angle')
plt.legend()
plt.savefig(os.path.join(args.output, 'pa_realizations.png'), bbox_inches='tight')
plt.close()
