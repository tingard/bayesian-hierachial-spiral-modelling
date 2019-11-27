import os
import shutil
import numpy as np
from pandas import Series, DataFrame
import pymc3 as pm
import theano.tensor as tt


class BHSM():
    def __init__(self, galaxies, build=True):
        """Accepts a list of groups of arm polar coordinates, and builds a
        PYMC3 hierarchial model to infer global distributions of pitch angle
        """
        self.galaxies = galaxies
        self.gal_n_arms = [len(g) for g in galaxies]
        self.n_arms = sum(self.gal_n_arms)
        # map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is
        # from the 3rd galaxy)
        self.gal_arm_map = np.concatenate([
            np.tile(i, n) for i, n in enumerate(self.gal_n_arms)
        ])
        # Create an array containing needed information in a stacked form
        self.data = DataFrame(
            np.concatenate([
                np.stack((
                    arm_T,
                    arm_R,
                    np.tile(sum(self.gal_n_arms[:gal_n]) + arm_n, len(arm_T)),
                    np.tile(gal_n, len(arm_T))
                ), axis=-1)
                for gal_n, galaxy in enumerate(galaxies)
                for arm_n, (arm_T, arm_R) in enumerate(galaxy)
            ]),
            columns=('theta', 'r', 'arm_index', 'galaxy_index')
        )
        # ensure correct dtypes
        self.data[['arm_index', 'galaxy_index']] = \
            self.data[['arm_index', 'galaxy_index']].astype(int)

        # assert we do not have any NaNs
        if np.any(self.data.isna()):
            raise ValueError('NaNs present in arm values')

        # ensure the arm indexing makes sense
        assert np.all(
            (
                np.unique(self.data['arm_index'])
                - np.arange(sum(self.gal_n_arms))
            ) == 0
        )
        if build:
            self.build_model()
        else:
            self.model = None

    def build_model(self, name='normal_model', prefix=''):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            # Global mean pitch angle
            self.mu_phi_scaled = pm.LogitNormal(
                'mu_phi_scaled', mu=0, sigma=1.5, testval=0.29
            )
            self.mu_phi = pm.Deterministic(
                'mu_phi', 90 * self.mu_phi_scaled
            )
            # inter-arm dispersion
            self.sigma_phi = pm.InverseGamma(
                'sigma_phi', alpha=2, beta=20, testval=5
            )
            # arm offset parameter
            self.c = pm.Cauchy(
                'c', alpha=0, beta=10, shape=self.n_arms,
                testval=np.tile(0, self.n_arms)
            )
            # radial noise
            self.sigma_r = pm.InverseGamma('sigma_r', alpha=2, beta=0.5)
            self.phi_arm = pm.TruncatedNormal(
                'phi_arm',
                mu=self.mu_phi, sd=self.sigma_phi,
                lower=0, upper=90, shape=self.n_arms,
            )
            # convert to a gradient for a linear fit
            self.b = tt.tan(np.pi / 180 * self.phi_arm)
            r = tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            )
            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )

    def do_inference(self, draws=1000, tune=500, target_accept=0.85,
                     max_treedepth=20, init='advi+adapt_diag',
                     **kwargs):
        if self.model is None:
            self.build_model()

        # it's important we now check the model specification, namely do we
        # have any problems with logp being undefined?
        with self.model as model:
            print(model.check_test_point())

        # Sampling
        with self.model as model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                max_treedepth=20,
                init=init,
                **kwargs
            )
        return trace

    def do_advi(self):
        if self.model is None:
            self.build_model()

        # it's important we now check the model specification, namely do we
        # have any problems with logp being undefined?
        with self.model as model:
            print(model.check_test_point())

        # Sampling
        with self.model as model:
            mean_field = pm.fit(
                method='advi',
                callbacks=[
                    pm.callbacks.CheckParametersConvergence(
                        diff='absolute'
                    )
                ]
            )
        return mean_field


# for the posterior predictive comparison
class CotUniformBHSM(BHSM):
    def build_model(self, name='cot_normal_model'):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            self.cot_phi = pm.Uniform(
                'cot_phi_gal',
                lower=1, upper=4,
                shape=self.n_arms
            )
            self.phi_gal = pm.Deterministic(
                'phi_gal', 180 / np.pi * tt.arctan(1 / self.cot_phi)
            )
            # note we don't model inter-galaxy dispersion here
            # intra-galaxy dispersion
            self.sigma_gal = pm.InverseGamma(
                'sigma_gal',
                alpha=2, beta=20, testval=5
            )
            # arm offset parameter
            self.c = pm.Cauchy(
                'c',
                alpha=0, beta=10,
                shape=self.n_arms,
                testval=np.tile(0, self.n_arms)
            )

            # radial noise
            self.sigma_r = pm.InverseGamma('sigma_r', alpha=2, beta=0.5)

            # Define Dependent variables
            self.phi_arm = pm.TruncatedNormal(
                'phi_arm',
                mu=self.phi_gal[self.gal_arm_map], sd=self.sigma_gal,
                lower=0, upper=90,
                shape=self.n_arms
            )

            # convert to a gradient for a linear fit
            self.b = tt.tan(np.pi / 180 * self.phi_arm)
            r = tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            )

            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )


def get_gal_pas(trace, galaxies, gal_arm_map, name=''):
    if type(galaxies) is not Series:
        galaxies = Series(galaxies)
    if name == '':
        arm_pas = trace[f'phi_arm']
    else:
        arm_pas = trace[f'{name}_phi_arm']
    gal_pas = Series(index=galaxies.index, dtype=object)
    for i, j in enumerate(gal_pas.index):
        arm_mask = gal_arm_map == j
        weights = list(map(lambda l: l.shape[1], galaxies.loc[j]))
        assert len(weights) == sum(arm_mask.astype(int))
        gal_pas.loc[j] = np.average(
            arm_pas.T[arm_mask],
            weights=weights,
            axis=0,
        )
    gal_pas = gal_pas.apply(Series)
    gal_pas.columns = gal_pas.columns.rename('sample')
    gal_pas.index = gal_pas.index.rename('galaxy')
    return gal_pas
