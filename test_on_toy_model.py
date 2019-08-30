import os
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
from run_hierarchial_model import run, run_advi, make_X_all
import argparse

parser = argparse.ArgumentParser(
    description=(
        'Hierarchially model galaxy pitch angle for a toy sample'
    )
)
parser.add_argument('--advi', action='store_true')
parser.add_argument('--outfolder', '-o', metavar='/path/to/trace/output',
                    type=str, default=None,
                    help='path to save traces (if desired)')

args = parser.parse_args()

if args.outfolder is not None:
    if not os.path.isfile(args.outfolder):
        os.mkdir(args.outfolder)

galaxies = pd.read_pickle('test-arms_s1=10_s2=3.pickle')

X_all = make_X_all(galaxies)

if args.advi:
    model, advi, tracker = run_advi(X_all, sample_size=None)
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
    # plt.figure()
    pm.plot_posterior(
        advi.approx.sample(3000),
        var_names=('mu_global', 'sd_global', 'sd_gal', 'sigma'),
        color='LightSeaGreen'
    )
    plt.show()
else:
    model, trace = run(
        X_all,
        sample_size=None,
        outfolder=args.outfolder,
    )

    pm.plot_posterior(
        trace.sample(3000),
        var_names=('mu_global', 'sd_global', 'sd_gal', 'sigma'),
        color='LightSeaGreen'
    )
    plt.show()
