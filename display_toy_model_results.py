import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
from bkp_hierarch import run, make_X_all, define_model

galaxies = pd.read_pickle('test-arms_s1=10_s2=3.pickle')

X_all = make_X_all(galaxies.iloc[:10])

with define_model(X_all) as model:
    trace = pm.load_trace('trace__toy_model')

pm.traceplot(trace, var_names=('mu_global', 'sd_global', 'sd_gal', 'sigma',))
plt.savefig('traceplot.png', bbox_inches='tight')
