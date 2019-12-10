import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize


def cot(phi):
    return 1 / np.tan(np.radians(phi))


def acot(a):
    return np.degrees(np.arctan(1 / a))


spiral_arms = pd.read_pickle('lib/unweighted_spirals.pickle')

# keep only galaxies with one arm or more
spirals_arms = spiral_arms[spiral_arms.notna().any(axis=1)]
# We want to scale r to have unit variance
# get all the radial points and calculate their std
normalization = np.concatenate(
    spirals_arms.drop('pipeline', axis=1).T.unstack().dropna().apply(lambda a: a.R).values
).std()

arm_pas = pd.Series([
    [arm.pa for arm in galaxy.dropna()]
    for _, galaxy in spirals_arms.drop('pipeline', axis=1).iterrows()
], index=spirals_arms.index)

# restrict to galaxies with pitch angles between acot(4) and acot(1)
gal_pas = spirals_arms.apply(
    lambda row: row['pipeline'].get_pitch_angle(row.dropna().values[1:])[0],
    axis=1
).reindex_like(arm_pas)

cot_mask = (gal_pas > acot(4)) & (gal_pas < acot(1))
arm_pas = arm_pas[cot_mask]

dataset = gal_pas[cot_mask]

# fit each possible distribution
p_norm = st.truncnorm.fit(
    dataset,
    (acot(4)-15) / 10, (acot(1) - 15) / 10, loc=20, scale=10
)
p_uniform = st.uniform.fit(dataset)


def truncated_normal(mu, sigma):
    return st.truncnorm(
        (acot(4) - mu) / sigma,  # lower bound
        (acot(1) - mu) / sigma,  # upper bound
        loc=mu,  # mean
        scale=sigma,  # std
    )


def f(p):
    return st.kstest(dataset, truncated_normal(*p).cdf).statistic


mu_phi, sigma_phi = minimize(f, (12, 15))['x']

plt.figure(figsize=(12, 8), dpi=100)
sns.distplot(dataset, kde=False, norm_hist=True, bins='scott',
             label='Galaxy Pitch angle')
x = np.linspace(acot(4), acot(1), 1000)
l = f'$\phi \sim \mathrm{{TruncatedNormal}}(\mu_\phi={mu_phi:.2f}, \sigma_\phi={sigma_phi:.2f})$'
plt.plot(x, truncated_normal(mu_phi, sigma_phi).pdf(x), label=l)
hist, bin_edges = np.histogram(
    acot(st.uniform(1, 4).rvs(int(1E6))),
    bins='scott', density=True
)
plt.step(bin_edges.tolist(), [0] + hist.tolist(),
         label=r'$\cot(\phi) \sim \mathrm{Uniform}(1, 4)$')
plt.plot(x, st.uniform(*p_uniform).pdf(x), label=r'$\phi \sim \mathrm{Uniform}(\cot^{-1}(4), \cot^{-1}(1))$')
plt.axvline(acot(4), c='k', alpha=0.2)
plt.axvline(acot(1), c='k', alpha=0.2)
plt.legend()

p_norm = st.kstest(dataset, truncated_normal(mu_phi, sigma_phi).cdf).pvalue
p_uniform = st.kstest(dataset, st.uniform(*p_uniform).cdf).pvalue
p_cot_uniform = st.kstest(cot(dataset), st.uniform(1, 4).cdf).pvalue
print(f'p(Normal) = {p_norm:.4f}')
print(f'p(Uniform) = {p_uniform:.4f}')
print(f'p(Uniform in Cot) = {p_cot_uniform:.4f}')
plt.savefig('cot_uniform_model_comparison.pdf', bbox_inches='tight')
