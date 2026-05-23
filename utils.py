import numpy as np


def _truncated_gaussian_pmf(mu, sigma, support):
    logp = -0.5 * ((support - mu) / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def generate_discrete_gaussian(mu, sigma, rng=(0, 100), N=512, seed=None):
    '''Draw N samples from a truncated discrete Gaussian over integers in [rng[0], rng[1]].

    PMF = normalized continuous N(mu, sigma) density evaluated at each integer
    in the support — mass outside the range is redistributed proportionally.
    '''
    support = np.arange(rng[0], rng[1] + 1)
    p = _truncated_gaussian_pmf(mu, sigma, support)
    gen = np.random.default_rng(seed)
    return gen.choice(support, size=N, p=p)


def generate_discrete_uniform(rng=(0, 100), N=512, seed=None):
    '''Draw N samples uniformly from integers in [rng[0], rng[1]] (inclusive).'''
    gen = np.random.default_rng(seed)
    return gen.integers(rng[0], rng[1] + 1, size=N)


def generate_discrete_bimodal_gaussian(mu1, sigma1, mu2, sigma2, w=0.5, rng=(0, 100), N=512, seed=None):
    '''Draw N samples from w * N(mu1, sigma1) + (1 - w) * N(mu2, sigma2),
    each component independently truncated and renormalized over integers in [rng[0], rng[1]].

    Note: components are normalized *before* mixing, so w stays the effective
    mixing weight even when one mode sits near the edge of the support.
    '''
    support = np.arange(rng[0], rng[1] + 1)
    p = w * _truncated_gaussian_pmf(mu1, sigma1, support) + (1 - w) * _truncated_gaussian_pmf(mu2, sigma2, support)
    gen = np.random.default_rng(seed)
    return gen.choice(support, size=N, p=p)

