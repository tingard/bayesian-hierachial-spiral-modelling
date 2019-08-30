import numpy as np
import pandas as pd
from gzbuilder_analysis.spirals import xy_from_r_theta
from gzbuilder_analysis.spirals.oo import Pipeline
from tqdm import trange


T = np.linspace(np.pi/2, 2*np.pi, 50)


def log_spiral(a, phi, t=T):
    return a * np.exp(np.tan(np.deg2rad(phi)) * t)


def gen_noisy_arms(phi, offset, r_sigma=(1E-1, 1E-1), t_sigma=0.1, n=15):
    rs = (
        log_spiral(1, phi, t=T)
        # systematic bias
        + np.expand_dims(
            np.random.randn(n) * r_sigma[0],
            1
        ) * np.ones((n, len(T)))
        # noise
        + np.random.randn(n, len(T)) * r_sigma[1]
    )
    rs *= 200 / np.max(rs)
    ts = np.tile(T + offset, n).reshape(n, -1)
    t_noise = np.random.randn(*ts.shape) * t_sigma
    xs, ys = xy_from_r_theta(rs, ts + t_noise)
    return np.stack((xs, ys), axis=-1)


def make_test_galaxy(n_arms, pa, sigma_pa, image_size=512):
    pas = np.random.randn(n_arms) * sigma_pa + pa
    base_theta = np.random.random() * 360
    arms = [
        gen_noisy_arms(
            pas[i],
            2*np.pi * i / n_arms + base_theta
        ) + image_size / 2
        for i in range(n_arms)
    ]
    return arms


def create_sample(N=100, pa_mu=20, inter_galaxy_sigma=10, intra_galaxy_sigma=3,
                  mean_n_arms=2):
    arms_df = pd.Series([]).rename('arms')
    arm_ns = np.random.poisson(mean_n_arms-1, size=N) + 1
    gal_pas = np.random.randn(N) * inter_galaxy_sigma + pa_mu
    with trange(N) as bar:
        for i in bar:
            gal = make_test_galaxy(arm_ns[i], gal_pas[i],
                                   intra_galaxy_sigma)
            drawn_arms = [line for arm in gal for line in arm]
            if len(drawn_arms) > 0:
                p = Pipeline([line for arm in gal for line in arm])
                arms = p.get_arms()
                arms_df.loc[i] = arms
    return arms_df


sigma_global = 10
sigma_gal = 3
df = create_sample(
    N=50,
    inter_galaxy_sigma=sigma_global,
    intra_galaxy_sigma=sigma_gal
)

df.to_pickle(
    'test-arms_inter={:.4f}_intra={:.4f}.pickle'.format(
        sigma_global,
        sigma_gal
    )
)
