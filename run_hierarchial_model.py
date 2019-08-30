import os
import shutil
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder


# TODO: rescale parameters
# TODO: deproject coordinates used!

def make_X_all(galaxies):
    """Creates an array of (theta, R, weight, galaxy_index, arm_index)
    for each point in each arm of each galaxy in the provided DataFrame.

    galaxies must be a DataFrame of `gzbuilder_analysis.spirals.oo.Arm` objects
    """
    out = []
    with tqdm(
        enumerate(galaxies.values),
        total=len(galaxies),
        desc='Calculating X_all'
    ) as bar:
        out = []
        for gal_no, galaxy in bar:
            for arm_no, arm in enumerate(galaxy):
                npoints = len(arm.R)
                if npoints > 10 and not arm.FLAGGED_AS_BAD:
                    arm_data = np.stack(
                        (
                            (arm.t * arm.chirality),
                            arm.R,
                            arm.point_weights,
                            np.tile(gal_no, npoints),
                            np.tile(
                                gal_no * len(galaxies) + arm_no,
                                npoints
                            )
                        ),
                        axis=1
                    )
                    out.append(arm_data)
    return np.concatenate(out)


def clean_and_sample(X_all, sample_size):
    X_masked = X_all[X_all.T[2] > 0]

    # optionally reduce the sample size to improve speed
    if sample_size is not None:
        if type(sample_size) == float and sample_size < 1:
            sample_size = int(len(X_all) * sample_size)
        sample_mask = np.random.choice(
            len(X_masked),
            size=int(sample_size),
            replace=False
        )
        X = X_masked[sample_mask]
    else:
        X = X_masked[:]
    return X


def get_encodings(X):
    enc = OrdinalEncoder(dtype=np.int32)
    gal_idx, arm_idx = enc.fit_transform(X[:, [3, 4]]).T
    n_arms = [
        len(np.unique(arm_idx[gal_idx == i])) for i in np.unique(gal_idx)
    ]
    arm_gal_idx = np.array([i for i, j in enumerate(n_arms) for k in range(j)])
    return gal_idx, arm_idx, arm_gal_idx


def define_model(X):
    """Define a hierarchial bayesian model for galaxy population pitch angle
    analysis. Accepts X, an array containing:

    X_i = (theta, R, weight, galaxy_index, arm_index)_i

    which can be created from a DataFrame of
    `gzbuilder_analysis.spirals.oo.Arm` objects using the `make_X_all` function
    """
    t, R, point_weights = X.T[:3]

    gal_idx, arm_idx, arm_gal_idx = get_encodings(X)

    n_gals = len(np.unique(gal_idx))
    n_unique_arms = len(np.unique(arm_idx))

    n_arms = [
        len(np.unique(arm_idx[gal_idx == i])) for i in np.unique(gal_idx)
    ]
    arm_gal_idx = np.array([i for i, j in enumerate(n_arms) for k in range(j)])
    with pm.Model() as model:
        # mu_global = pm.Uniform('mu_global', lower=3, upper=80, testval=15)
        mu_global = pm.Uniform('mu_global', lower=5, upper=60, testval=20)

        # measures inter-galaxy pitch angle dispersion
        sd_global = pm.HalfCauchy('sd_global', beta=10, testval=3)
        # pm.InverseGamma('sd_global', alpha=2, beta=10, testval=10)

        # measures intra-galaxy pitch angle dispersion
        sd_gal = pm.HalfCauchy('sd_gal', beta=10, testval=3)

        # # Centered version:
        # psi_gal = pm.Normal('psi_gal',
        #                     mu=mu_global, sd=sd_global, shape=n_gals)
        # psi_arm = pm.Normal('psi_arm',
        #                     mu=psi_gal[gal_idx], sd=sd_gal[gal_idx])

        # Non-centred version:
        psi_gal_offset = pm.Normal('psi_gal_offset', mu=0, sd=1, shape=n_gals)
        psi_gal = pm.Deterministic(
            'psi_gal',
            mu_global + psi_gal_offset * sd_global
        )

        psi_arm_offset = pm.Normal('psi_arm_offset', mu=0, sd=1,
                                   shape=n_unique_arms)
        psi_arm = pm.Deterministic(
            'psi_arm',
            psi_gal[arm_gal_idx] + psi_arm_offset * sd_gal
        )

        a_arm = pm.Uniform(
            'a',
            lower=1E-2, upper=20,
            testval=1, shape=n_unique_arms
        )

        # Define our expecation of R (note the conversion to radians of pitch
        # angle)
        r_est = (
            a_arm[arm_idx]
            * theano.tensor.exp(
                theano.tensor.tan(psi_arm[arm_idx] * np.pi / 180)
                * t
            )
        )

        sigma_r = pm.HalfCauchy('100_sigma', beta=3, testval=1) / 100

        pm.Normal('L', mu=r_est, sd=sigma_r, observed=R)
    return model


def print_info(X):
    # for logging
    print('{} galaxies, {} spiral arms, {} points'.format(
        len(np.unique(X[:, 3])), len(np.unique(X[:, 4])), len(X)
    ))


def run_advi(X_all, sample_size=None, **kwargs):
    X = clean_and_sample(X_all, sample_size)

    # for logging
    print_info(X)
    # X_batched = pm.Minibatch(X, 100)
    with define_model(X) as model:
        assert np.isfinite(model.logp(model.test_point))
        advi = pm.ADVI()
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # callable that returns mean
            std=advi.approx.std.eval  # callable that returns std
        )
        param_check = pm.callbacks.CheckParametersConvergence(
            diff='absolute'
        )
        try:
            advi.fit(20000, callbacks=[tracker, param_check])
        except FloatingPointError:
            pass
    return model, advi, tracker


def run(X_all, sample_size=None, outfolder=None):
    X = clean_and_sample(X_all, sample_size)

    # for logging
    print_info(X)

    with define_model(X) as model:
        trace = pm.sample(1000, tune=1000, target_accept=0.90)

    if outfolder is not None:
        try:
            os.mkdir(outfolder)
        except FileExistsError:
            shutil.rmtree(outfolder)
        pm.save_trace(trace, directory=outfolder, overwrite=True)

    return model, trace


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Hierarchially model galaxy pitch angle for a sample of '
            'subject_ids'
        )
    )
    parser.add_argument('--subjects', '-s',
                        metavar='/path/to/subject-id-list.csv',
                        default=None, type=str,
                        help='Location of subject ID csv (newline-separated)')
    parser.add_argument('--npoints', '-n', metavar='N',
                        type=int, default=None,
                        help='path to save traces (if desired)')
    parser.add_argument('--outfolder', '-o', metavar='/path/to/trace/output',
                        type=str, default=None,
                        help='path to save traces (if desired)')

    args = parser.parse_args()

    agg_results = pd.read_pickle('lib/aggregation_results.pickle')

    if args.subjects is None:
        sid_list = agg_results.index.values
    else:
        sid_list = np.loadtxt(args.subjects, dtype=np.int64)

    X_all = make_X_all(
        agg_results.Arms.loc[sid_list]
    )

    if not os.path.isfile(args.outfolder):
        os.mkdir(args.outfolder)

    run(X_all=X_all, sample_size=args.npoints, outfolder=args.outfolder)
