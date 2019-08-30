import numpy as np
import scipy.stats as st

base_t = np.linspace(0, np.pi, 500)


def log_spiral(t, phi):
    return np.exp(np.tan(np.deg2rad(phi)) * t)


def gen_noisy_arm(phi, N=500, t_offset=None,
                  t_noise_amp=1E-4, r_noise_amp=3E-2):
    t = np.linspace(0, np.pi, N)
    r = log_spiral(t, phi)
    if t_offset is None:
        t_offset = np.random.random() * 2 * np.pi

    t_noise = np.random.normal(loc=0, scale=t_noise_amp, size=t.shape)
    r_noise = np.random.normal(loc=0, scale=r.max() * r_noise_amp,
                               size=r.shape)

    T = t + t_offset + t_noise
    R = r + r_noise
    return T, R


def gen_galaxy(n_arms, pa, sigma_pa, pa_bounds=(0.1, 60), **kwargs):
    pas = st.truncnorm.rvs(*pa_bounds, loc=pa, scale=sigma_pa, size=n_arms)
    base_offset = np.random.random() * 2 * np.pi
    return [
        gen_noisy_arm(
            pas[i],
            t_offset=2*np.pi * i / n_arms + base_offset,
            **{'N': 500, **kwargs}
        )
        for i in range(n_arms)
    ]
