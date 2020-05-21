import os
import sys
import shutil
import numpy as np
from pandas import Series, DataFrame
import pymc3 as pm
import theano.tensor as tt


# Assume Uniform prior on galaxy pitch angle, and arm pitch angle normally
# distributed around a group mean
class BHSM():
    def __init__(self, galaxies, build=True):
        """Accepts a list of groups of arm polar coordinates, and builds a
        PYMC3 hierarchial model to infer global distributions of pitch angle
        """
        self.galaxies = galaxies
        self.gal_n_arms = gal_n_arms = galaxies.apply(len)
        self.n_arms = self.gal_n_arms.sum()
        # map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is
        # from the 3rd galaxy)
        self.gal_arm_map = np.concatenate([
            np.tile(i, n) for i, n in enumerate(self.gal_n_arms.values)
        ])
        # Create an array containing needed information in a stacked form
        self.data = DataFrame(
            np.concatenate([
                np.stack((
                    arm_T,
                    arm_R,
                    np.tile(sum(self.gal_n_arms.iloc[:gal_n]) + arm_n, len(arm_T)),
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

        self.point_arm_map = self.data['arm_index'].values
        # assert we do not have any NaNs
        if np.any(self.data.isna()):
            raise ValueError('NaNs present in arm values')

        # ensure the arm indexing makes sense
        indexing_sensible = np.all(
            (
                np.unique(self.data['arm_index'])
                - np.arange(self.n_arms)
            ) == 0
        )
        assert indexing_sensible, 'Something went wrong with arm indexing'

        if build is True:
            self.build_model()
        else:
            self.model = None

    def build_model(self, name=''):
        """Function to be overwritten by specific subclass of BHSM
        """
        pass

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
        raise NotImplementedError('ADVI is not implemented')


class UniformBHSM(BHSM):
    def build_model(self, name=''):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            # Global mean pitch angle
            self.phi_gal = pm.Uniform(
                'phi_gal',
                lower=0, upper=90,
                shape=len(self.galaxies)
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
            r = pm.Deterministic('r', tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            ))

            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )


class HierarchialNormalBHSM(BHSM):
    def build_model(self, name='normal_model'):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            # Global mean pitch angle
            self.mu_phi = pm.Uniform(
                'mu_phi',
                lower=0, upper=90
            )
            self.sigma_phi = pm.InverseGamma(
                'sigma_phi', alpha=2, beta=15, testval=8
            )
            self.sigma_gal = pm.InverseGamma(
                'sigma_gal', alpha=2, beta=15, testval=8
            )
            # define a mean galaxy pitch angle
            self.phi_gal = pm.TruncatedNormal(
                'phi_gal',
                mu=self.mu_phi, sd=self.sigma_phi,
                lower=0, upper=90, shape=len(self.galaxies),
            )
            # draw arm pitch angles centred around this mean
            self.phi_arm = pm.TruncatedNormal(
                'phi_arm',
                mu=self.phi_gal[self.gal_arm_map], sd=self.sigma_gal,
                lower=0, upper=90,
                shape=len(self.gal_arm_map),
            )
            # convert to a gradient for a linear fit
            self.b = tt.tan(np.pi / 180 * self.phi_arm)
            # arm offset parameter
            self.c = pm.Cauchy(
                'c', alpha=0, beta=10, shape=self.n_arms,
                testval=np.tile(0, self.n_arms)
            )
            # radial noise
            self.sigma_r = pm.InverseGamma('sigma_r', alpha=2, beta=0.5)
            r = pm.Deterministic('r', tt.exp(
                self.b[self.point_arm_map] * self.data['theta']
                + self.c[self.point_arm_map]
            ))
            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )


# for the posterior predictive comparison
class CotUniformBHSM(BHSM):
    def build_model(self, name='cot_uniform_model'):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            self.cot_phi = pm.Uniform(
                'cot_phi_gal',
                lower=1, upper=4,
                shape=len(self.galaxies)
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
            r = pm.Deterministic('r', tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            ))

            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )


class ArchimedianBHSM(BHSM):
    r"""This model fits archimedian spirals, rather than the logarithmic spirals
    present in other models. There is no assumed relationship between arms.

    An Archimedian spiral is given by

    $$r = a\theta^{1/n}$$

    Here we constrain $n$ to be the same for all arms, and add a rotation
    paramter $\delta\theta$

    $$r = a\left(\theta + \delta\theta\right)^{m}$$

    """

    def build_model(self, n=None, name='archimedian_model'):
        with pm.Model(name=name) as self.model:
            if n is None:
                # one n per galaxy, or per arm?
                self.n_choice = pm.Categorical(
                    'n_choice',
                    [1, 1, 0, 1, 1],
                    testval=1,
                    shape=len(self.galaxies)
                )
                self.n = pm.Deterministic('n', self.n_choice - 2)
                self.chirality_correction = tt.switch(self.n < 0, -1, 1)
            else:
                msg = 'Parameter $n$ must be a nonzero float'
                try:
                    n = float(n)
                except ValueError:
                    pass
                finally:
                    assert isinstance(n, float) and n != 0, msg

                self.n_choice = None
                self.n = pm.Deterministic('n', np.repeat(n, len(self.galaxies)))

            self.chirality_correction = tt.switch(self.n < 0, -1, 1)
            self.a = pm.HalfCauchy(
                'a',
                beta=1, testval=1,
                shape=self.n_arms
            )
            self.psi = pm.Normal(
                'psi',
                mu=0, sigma=1, testval=0.1,
                shape=self.n_arms,
            )
            self.sigma_r = pm.InverseGamma(
                'sigma_r',
                alpha=2, beta=0.5
            )
            # Unfortunately, as we need to reverse the theta points for arms
            # with n < 1, and rotate all arms to start at theta = 0,
            # we need to do some model-mangling
            self.t_mins = Series({
                i: self.data.query('arm_index == @i')['theta'].min()
                for i in np.unique(self.data['arm_index'])
            })
            r_stack = [
                self.a[i]
                * tt.power(
                    (
                        self.data.query('arm_index == @i')['theta'].values
                        - self.t_mins[i]
                        + self.psi[i]
                    ),
                    1 / self.n[int(self.gal_arm_map[i])]
                )[::self.chirality_correction[int(self.gal_arm_map[i])]]
                for i in np.unique(self.data['arm_index'])
            ]
            r = pm.Deterministic('r', tt.concatenate(r_stack))
            self.likelihood = pm.StudentT(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'].values,
            )

    def do_inference(self, draws=20000, tune=2000, init='adapt_diag', **kwargs):
        if self.model is None:
            self.build_model()

        # it's important we now check the model specification, namely do we
        # have any problems with logp being undefined?
        with self.model as model:
            test_point = model.check_test_point()

            if len(self.model.name) > 0:
                l_key = self.model.name + '_'
            else:
                l_key = ''

        print(test_point)
        if np.isnan(test_point['{}Likelihood'.format(l_key)]):
            print('The model\'s test point had an undefined likelihood, meaning sampling will fail')
            sys.exit(0)

        # Sampling
        with self.model as model:
            if self.n_choice is not None:
                step1 = pm.CategoricalGibbsMetropolis(self.n_choice)
                step2 = pm.Metropolis([self.a, self.psi, self.sigma_r])
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    init=init,
                    step=[step1, step2],
                    **kwargs
                )
            else:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    init=init,
                    **kwargs
                )
        return trace


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
