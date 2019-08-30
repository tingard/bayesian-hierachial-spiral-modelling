import os
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
from sample_generation import gen_galaxy


n_arms = 20  # np.random.poisson(0.9) + 2
galaxy = gen_galaxy(n_arms, 15, 5, N=250)

point_data = np.concatenate([
    np.stack((arm_T, arm_R, np.tile(arm_n, len(arm_T))), axis=-1)
    for arm_n, (arm_T, arm_R) in enumerate(galaxy)
])

R, T, arm_idx = point_data.T
arm_idx = arm_idx.astype(int)

with pm.Model() as model:
    # model r = a * exp(tan(phi) * t) as log(r) = b * t + c,
    # but have a uniform prior on phi rather than the gradient!
    gal_pa_mu = pm.Uniform('pa', lower=0.1, upper=60, testval=20)
    gal_pa_sd = pm.HalfCauchy('pa_sd', beta=10, testval=10)

    # we want this:
    # arm_pa = pm.TruncatedNormal(
    #     'arm_pa',
    #     mu=gal_pa_mu, sd=gal_pa_sd,
    #     lower=0.1, upper=60,
    #     shape=n_arms,
    # )
    # Specified in a non-centred way:
    arm_pa_mu_offset = pm.Normal(
        'arm_pa_mu_offset',
        mu=0, sd=1, shape=n_arms
    )
    arm_pa = pm.Deterministic(
        'arm_pa',
        gal_pa_mu + gal_pa_sd * arm_pa_mu_offset
    )
    pm.Potential(
        'lower_arm_pa_bound',
        tt.switch(tt.all(arm_pa > 0.1), 0, -np.inf)
    )
    pm.Potential(
        'upper_arm_pa_bound',
        tt.switch(tt.all(arm_pa < 60), 0, -np.inf)
    )

    arm_b = pm.Deterministic('b', tt.tan(np.pi / 180 * arm_pa))
    arm_c = pm.Normal('c', mu=0, sigma=20, shape=n_arms)

    sigma = pm.HalfCauchy('sigma', beta=1)

    likelihood = pm.Normal(
        'y',
        mu=tt.exp(arm_b[arm_idx] * T + arm_c[arm_idx]),
        sigma=sigma,
        observed=R
    )

    trace = pm.sample(2000, tune=1000, target_accept=0.95)

# display the total number and percentage of divergent
divergent = trace['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent %.1f' % divperc)

loc = os.path.abspath(os.path.dirname(__file__))

# Save a traceplot
pm.traceplot(trace, var_names=('pa', 'pa_sd', 'sigma'))
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy_traceplot.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(trace, var_names=('pa', 'pa_sd', 'sigma'))
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy_posterior.png'),
    bbox_inches='tight'
)
plt.close()

# plot the "galaxy" used
for (_t, _r) in galaxy:
    plt.plot(*xy_from_r_theta(_r, _t), '.')
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy.png'),
    bbox_inches='tight'
)
