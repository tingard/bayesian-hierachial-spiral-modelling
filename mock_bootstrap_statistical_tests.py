import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import warnings

warnings.simplefilter('ignore', UserWarning)

N_POINTS = 50
N_OBS = 100
SIGMA = 1
TRUE_RV = st.t(3)
TEST_RV = st.t(3)

# create N_POINTS points drawn from a known distribution
# each of which has N_OBS observations
raw_data = np.ones((N_OBS, N_POINTS)) * TRUE_RV.rvs(N_POINTS)

# add some noise to this data
data = raw_data + st.norm(scale=SIGMA).rvs((N_OBS, N_POINTS))

# test for each observation
ks = pd.Series([], name='ks_test_result')
anderson = pd.Series([], name='anderson_test_result')
test_dataset = TEST_RV.rvs(10000)
with trange(data.shape[0]) as bar:
    for i in bar:
        sample = data[i]
        anderson[i] = st.anderson_ksamp((
            sample, test_dataset,
        ))
        ks[i] = st.kstest(
            sample, TEST_RV.cdf,
        )
ks = ks.apply(pd.Series).rename(columns={0: 'value', 1: 'p'})
anderson = anderson.apply(pd.Series).rename(columns={0: 'value', 1: 'levels', 2: 'significance'})

f, ax = plt.subplots(nrows=2, figsize=(12, 7))

plt.sca(ax[0])
plt.title('Anderson-Darling test results for posterior samples')
sns.kdeplot(anderson['value'], label='')
print('Thresholds:')
for i, j in zip(('25%', '10%', '5%', '2.5%', '1%'), np.stack(anderson['levels'].values).mean(axis=0)):
    freq = (anderson['value'] >= j).sum() / len(anderson['value'])
    print(f' {i: <4}: {j:.3f}, reject at this level {freq:.0%} of the time')
    plt.axvline(j, color='k', alpha=0.2)
    plt.text(j, plt.ylim()[1]*0.9, i)
plt.axvline(anderson['value'].mean(), color='r', ls='--', label='Expectation value')
plt.xlabel('Anderson-Darling statistic')
plt.sca(ax[1])
plt.title(r'Kolmogorov–Smirnov test results for posterior samples')
sns.kdeplot(np.log10(ks['p']), label='', shade=True)
plt.xticks(np.arange(-6, 1, 1), 10.**np.arange(-6, 1, 1))
plt.xlabel(r'Probability of being drawn from target distribution')
plt.tight_layout()
plt.show()
