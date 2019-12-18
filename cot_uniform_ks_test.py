import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
import generate_sample


def cot(phi):
    return 1 / np.tan(np.radians(phi))


def acot(a):
    return np.degrees(np.arctan(1 / a))


LOWER_COT_BOUND = 1.19
UPPER_COT_BOUND = 4.75

lower_phi_bound = acot(UPPER_COT_BOUND)
upper_phi_bound = acot(LOWER_COT_BOUND)
# read in the galaxy builder spiral arms dataset
spiral_arms = pd.read_pickle('lib/unweighted_spirals.pickle')

# use the generate_sample function for consistency with other methods. Restrict
# to galaxies with pitch angle between our bounds
sample_indices = generate_sample.generate_sample(None, None, lambda a: (a > lower_phi_bound) and (a < upper_phi_bound))

# get length-weighted pitch angles for our chosen sample
dataset = spiral_arms.loc[sample_indices.index].apply(
    lambda row: row['pipeline'].get_pitch_angle(row.dropna().values[1:])[0],
    axis=1
).dropna()

# fit each possible distribution
p_norm = st.truncnorm.fit(
    dataset,
    (lower_phi_bound - 15) / 10, (upper_phi_bound - 15) / 10, loc=20, scale=10
)

def truncated_normal(mu, sigma):
    return st.truncnorm(
        (lower_phi_bound - mu) / sigma,  # lower bound
        (upper_phi_bound - mu) / sigma,  # upper bound
        loc=mu,  # mean
        scale=sigma,  # std
    )

def f(p):
    return st.kstest(dataset, truncated_normal(*p).cdf).statistic

mu_phi, sigma_phi = minimize(f, (12, 15))['x']

plt.figure(figsize=(12, 8), dpi=100)
sns.distplot(dataset, kde=False, norm_hist=True, bins='scott',
             label='Galaxy Pitch angle')
x = np.linspace(lower_phi_bound, upper_phi_bound, 1000)
l = f'$\phi \sim \mathrm{{TruncatedNormal}}(\mu_\phi={mu_phi:.2f}, \sigma_\phi={sigma_phi:.2f})$'
plt.plot(x, truncated_normal(mu_phi, sigma_phi).pdf(x), label=l)
hist, bin_edges = np.histogram(
    acot(st.uniform(LOWER_COT_BOUND, UPPER_COT_BOUND - LOWER_COT_BOUND).rvs(int(1E6))),
    bins='scott', density=True
)
plt.step(bin_edges.tolist(), [0] + hist.tolist(),
         label=r'$\cot(\phi) \sim \mathrm{Uniform}'
               +f'({LOWER_COT_BOUND}, {UPPER_COT_BOUND})$')
plt.plot(x, st.uniform(lower_phi_bound, upper_phi_bound - lower_phi_bound).pdf(x),
         label=r'$\phi \sim \mathrm{Uniform}(\cot^{-1}(4), \cot^{-1}(1))$')
plt.axvline(lower_phi_bound, c='k', alpha=0.2)
plt.axvline(upper_phi_bound, c='k', alpha=0.2)
plt.legend()

significance_norm = st.kstest(dataset, truncated_normal(mu_phi, sigma_phi).cdf).pvalue
significance_uniform = st.kstest(dataset, st.uniform(lower_phi_bound, upper_phi_bound - lower_phi_bound).cdf).pvalue
significance_cot_uniform = st.kstest(cot(dataset), st.uniform(LOWER_COT_BOUND, UPPER_COT_BOUND - LOWER_COT_BOUND).cdf).pvalue
print(f'p(Normal) = {significance_norm:.4f}')
print(f'p(Uniform) = {significance_uniform:.4f}')
print(f'p(Uniform in Cot) = {significance_cot_uniform:.4f}')
plt.savefig('cot_uniform_model_comparison.pdf', bbox_inches='tight')
