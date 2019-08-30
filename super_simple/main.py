import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt


np.random.seed(0)

print('[1/4] Making sample')
N = 300
A_MU = 1
PSI_MU = 20
PSI_SIGMA = 4
R_SIGMA = 0.01
print('\tGlobal parameters: μ = {:.2f}, σ = {:.2f}'.format(
    PSI_MU, PSI_SIGMA
))


def logsp(t, a, phi):
    if len(t.shape) > 1:
        return np.exp(np.tan(np.deg2rad(phi)) * t.T + a).T
    return np.exp(np.tan(np.deg2rad(phi)) * t + a)


def xy_from_r_theta(r, theta, mux=0, muy=0):
    return np.stack((mux + r * np.cos(theta), muy + r * np.sin(theta)))


pa_dist = np.random.randn(N) * PSI_SIGMA + PSI_MU
pa_dist = np.clip(pa_dist, 5, 70)
a_dist = np.random.randn(N) * 0.1 + A_MU
offsets = np.random.random(N) * 2 * np.pi
t = np.linspace(0, np.pi, 100)

ts = np.tile(t, N).reshape(N, -1)
rs = logsp(ts, a_dist, pa_dist) + np.random.randn(*ts.shape) * R_SIGMA

_, ax = plt.subplots(ncols=2, figsize=(12, 4))
plt.sca(ax[0])
sns.kdeplot(pa_dist, shade=True)
lims = ax[0].get_ylim()
plt.vlines(pa_dist.mean(), 0, lims[1])
plt.hlines(0, pa_dist.mean() - pa_dist.std(), pa_dist.mean() + pa_dist.std(),
           linewidth=5)
plt.ylim(*lims)
ax[0].text(
    pa_dist.mean() + pa_dist.std() / 10,
    lims[1] / 20,
    '{:.4f}'.format(pa_dist.std())
)
plt.title('Input pitch angle distribution')
plt.sca(ax[1])
sns.kdeplot(a_dist, shade=True)
lims = ax[1].get_ylim()
plt.vlines(a_dist.mean(), 0, lims[1])
plt.ylim(*lims)
plt.title('Input offset distribution')
plt.savefig('super_simple/toy_model_target.png', bbox_inches='tight')
plt.close()

print('[2/4] Defining model')
X = np.array([
    [ts[i][j], rs[i][j], 1, i, 0]
    for i in range(N)
    for j in range(len(ts[i]))
])
t, R, point_weights = X.T[:3]
logR = np.log(R)
arm_idx = X[:, 3].astype(int)

with pm.Model() as model:
    # rather than model r = a * exp(tan(phi) * t)
    # change so that we're fitting the gradient and intercept of logR = bt + c
    # where b = tan(phi) and c = log(a)
    b_est = np.tan(np.rad2deg(PSI_MU))
    mu_b_gal = pm.HalfCauchy('mu_b_gal', beta=b_est, testval=b_est)
    sd_b_gal = pm.HalfCauchy('sd_b_gal', beta=1)

    mu_b_offset = pm.Normal('mu_b_offset', mu=0, sd=1,
                            shape=N)
    b_arm = pm.Deterministic(
        'b_arm',
        mu_b_gal + mu_b_offset * sd_b_gal
    )

    c_arm = pm.Normal('c_arm', mu=0, sd=10, shape=N)

    logr_est = b_arm[arm_idx] * t + c_arm[arm_idx]

    mu_phi = pm.Deterministic('mu_global', tt.tan(mu_b_gal))
    sd_phi = pm.Deterministic('sd_gal', tt.tan(sd_b_gal))

    sigma_logR = pm.HalfCauchy('10sigma_logR', 10, testval=10) / 10

    pm.Normal('L', mu=logr_est, sigma=sigma_logR, observed=logR)


# with pm.Model() as model:
#
#     mu_gal = pm.Uniform('mu_global', lower=5, upper=60, testval=PSI_MU)
#
#     # measures intra-galaxy pitch angle dispersion
#     sd_gal = pm.HalfCauchy('sd_gal', beta=15, testval=PSI_SIGMA)
#
#     # non-centred definition of `mu_arm`
#     mu_arm = pm.Normal('mu_arm', mu=mu_gal, sd=sd_gal, shape=N)
#     # mu_arm_offset = pm.Normal('mu_arm_offset', mu=0, sd=1,
#     #                           shape=N)
#     # mu_arm = pm.Deterministic(
#     #     'mu_arm',
#     #     mu_gal + mu_arm_offset * sd_gal
#     # )
#
#     # definition of `a` for each arm
#     a_arm = pm.HalfCauchy(
#         'a', beta=1, shape=N, testval=1
#     ) * A_MU
#
#     # Define our expecation of R (note the conversion to radians of pitch
#     # angle)
#     # np.exp(np.tan(np.deg2rad(phi)) * t.T + a).T
#     logr_est = tt.tan(mu_arm[arm_idx] * np.pi / 180) * t + a_arm[arm_idx]
#
#     sigma_logR = pm.HalfCauchy('10sigma_logR', 10, testval=10) / 10
#
#     pm.Normal('L', mu=logr_est, sigma=sigma_logR, observed=logR)
#     # pm.Normal('L', mu=r_est, sd=sigma_r, observed=R)


print('[3/4] Sampling')
with model:
    advi = pm.ADVI()
    # tracker = pm.callbacks.Tracker(
    #     mean=advi.approx.mean.eval,  # callable that returns mean
    #     std=advi.approx.std.eval  # callable that returns std
    # )
    approx = advi.fit(10000)  #, callbacks=[tracker])
    # trace = pm.sample(500, tune=800, target_accept=0.95)

print('[4/4] Making plots')
# fig = plt.figure(figsize=(16, 9))
# mu_ax = fig.add_subplot(221)
# std_ax = fig.add_subplot(222)
# hist_ax = fig.add_subplot(212)
# mu_ax.plot(tracker['mean'])
# mu_ax.set_title('Mean track')
# std_ax.plot(tracker['std'])
# std_ax.set_title('Std track')
# hist_ax.plot(advi.hist)
# hist_ax.set_title('Negative ELBO track')
# plt.savefig('super_simple/advi_tracker_plot.png', bbox_inches='tight')
# plt.close()
pm.plot_posterior(approx.sample(5000), var_names=('mu_global', 'sd_gal', '10sigma_logR'))
plt.savefig('super_simple/advi_trace_plot.png', bbox_inches='tight')

# pm.traceplot(trace, var_names=('mu_global', 'sd_gal', 'sigma_logR'))
