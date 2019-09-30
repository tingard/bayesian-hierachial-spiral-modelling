import numpy as np
import pymc3 as pm
import theano.tensor as tt
from types import SimpleNamespace


class BHSM():
    def __init__(self, galaxies):
        self.galaxies = galaxies
        self.gal_n_arms = [len(g) for g in galaxies]

        # map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is
        # from the 3rd galaxy)
        self.gal_arm_map = np.concatenate([
            np.tile(i, n) for i, n in enumerate(self.gal_n_arms)
        ])

        # Create an array containing needed information in a stacked form
        self.point_data = np.concatenate([
            np.stack((
                arm_T,
                arm_R,
                np.tile(sum(self.gal_n_arms[:gal_n]) + arm_n, len(arm_T)),
                np.tile(gal_n, len(arm_T))
            ), axis=-1)
            for gal_n, galaxy in enumerate(galaxies)
            for arm_n, (arm_T, arm_R) in enumerate(galaxy)
        ])

        self.T, self.R, arm_idx, self.gal_idx = self.point_data.T
        self.arm_idx = arm_idx.astype(int)

        # ensure the arm indexing makes sense
        assert np.all(
            (np.unique(arm_idx) - np.arange(sum(self.gal_n_arms))) == 0
        )
        self.build_model()

    def build_model(self):
        self.model = SimpleNamespace()
        # Define Stochastic variables
        with pm.Model() as self.model:
            # Global mean pitch angle
            self.global_pa_mu = pm.Uniform(
                'pa',
                lower=0, upper=90,
                testval=20
            )

            # inter-galaxy dispersion
            self.global_pa_sd = pm.InverseGamma('pa_sd', alpha=2, beta=20)

            # intra-galaxy dispersion
            self.gal_pa_sd = pm.InverseGamma('gal_pa_sd', alpha=2, beta=20)

            # arm offset parameter
            self.arm_c = pm.Cauchy(
                'c',
                alpha=0, beta=10,
                shape=len(self.gal_arm_map),
                testval=np.tile(0, len(self.gal_arm_map))
            )

            # radial noise
            self.sigma = pm.InverseGamma('sigma', alpha=2, beta=0.5)

        # Define Dependent variables
        with self.model:
            # we want this:
            # gal_pa_mu = pm.TruncatedNormal(
            #     'gal_pa_mu',
            #     mu=global_pa_mu, sd=global_pa_sd,
            #     lower=0.1, upper=60,
            #     shape=n_gals,
            # )
            # arm_pa = pm.TruncatedNormal(
            #     'arm_pa_mu',
            #     mu=gal_pa_mu[gal_arm_map], sd=gal_pa_sd[gal_arm_map],
            #     lower=0.1, upper=60,
            #     shape=len(gal_arm_map),
            # )
            # Specified in a non-centred way:

            self.gal_pa_mu_offset = pm.Normal(
                'gal_pa_mu_offset',
                mu=0, sd=1, shape=len(self.galaxies),
            )
            self.gal_pa_mu = pm.Deterministic(
                'gal_pa_mu',
                self.global_pa_mu + self.gal_pa_mu_offset * self.global_pa_sd
            )

            # use a Potential for the truncation, pm.Potential('foo', N) simply
            # adds N to the log likelihood
            pm.Potential(
                'gal_pa_mu_bound',
                (
                    tt.switch(tt.all(self.gal_pa_mu > 0), 0, -np.inf)
                    + tt.switch(tt.all(self.gal_pa_mu < 90), 0, -np.inf)
                )
            )

            self.arm_pa_mu_offset = pm.Normal(
                'arm_pa_mu_offset',
                mu=0, sd=1, shape=sum(self.gal_n_arms),
                testval=np.tile(0, sum(self.gal_n_arms))
            )
            self.arm_pa = pm.Deterministic(
                'arm_pa',
                self.gal_pa_mu[self.gal_arm_map]
                + self.arm_pa_mu_offset * self.gal_pa_sd
            )
            pm.Potential(
                'arm_pa_mu_bound',
                (
                    tt.switch(tt.all(self.arm_pa > 0), 0, -np.inf)
                    + tt.switch(tt.all(self.arm_pa < 90), 0, -np.inf)
                )
            )

            # convert to a gradient for a linear fit
            self.arm_b = tt.tan(np.pi / 180 * self.arm_pa)
            self.arm_r = tt.exp(
                self.arm_b[self.arm_idx] * self.T
                + self.arm_c[self.arm_idx]
            )
            pm.Potential(
                'arm_r_bound',
                tt.switch(tt.all(self.arm_r < 1E4), 0, -np.inf)
            )
            # likelihood function
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=self.arm_r,
                sigma=self.sigma,
                observed=self.R
            )
