from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from utils import (
    generate_discrete_bimodal_gaussian,
    generate_discrete_gaussian,
    generate_discrete_uniform,
)


@dataclass
class DistPriors:
    '''Priors over distribution configs. `DistStream` draws one config per item from these.'''
    types: tuple = ("UNIFORM", "GAUSSIAN", "BIMODAL")
    type_weights: tuple = (1.0, 2.0, 2.0)
    mu_range: tuple = (10.0, 90.0)
    sigma_range: tuple = (1.0, 15.0)
    w_range: tuple = (0.3, 0.7)
    bimodal_min_separation: float = 15.0


def sample_spec(priors, gen):
    '''Draw one distribution config from `priors`. Returns `{type, params}`.'''
    weights = np.array(priors.type_weights, dtype=float)
    weights /= weights.sum()
    dist_type = str(gen.choice(priors.types, p=weights))

    if dist_type == "UNIFORM":
        return {"type": "UNIFORM", "params": {}}

    if dist_type == "GAUSSIAN":
        return {
            "type": "GAUSSIAN",
            "params": {
                "mu": float(gen.uniform(*priors.mu_range)),
                "sigma": float(gen.uniform(*priors.sigma_range)),
            },
        }

    if dist_type == "BIMODAL":
        lo, hi = priors.mu_range
        min_sep = priors.bimodal_min_separation
        # Rejection sample until the two modes are at least `min_sep` apart,
        # so "bimodal" actually looks bimodal and the modality label stays meaningful.
        while True:
            mu1 = float(gen.uniform(lo, hi))
            mu2 = float(gen.uniform(lo, hi))
            if abs(mu1 - mu2) >= min_sep:
                break
        return {
            "type": "BIMODAL",
            "params": {
                "mu1": mu1,
                "sigma1": float(gen.uniform(*priors.sigma_range)),
                "mu2": mu2,
                "sigma2": float(gen.uniform(*priors.sigma_range)),
                "w": float(gen.uniform(*priors.w_range)),
            },
        }

    raise ValueError(f"unknown dist type: {dist_type!r}")


def _samples_from_spec(spec, N, gen):
    # Derive a per-call seed from the stream's RNG so seed-set means full reproducibility.
    seed = int(gen.integers(0, 2**31 - 1))
    if spec["type"] == "UNIFORM":
        return generate_discrete_uniform(N=N, seed=seed)
    if spec["type"] == "GAUSSIAN":
        return generate_discrete_gaussian(N=N, seed=seed, **spec["params"])
    if spec["type"] == "BIMODAL":
        return generate_discrete_bimodal_gaussian(N=N, seed=seed, **spec["params"])
    raise ValueError(f"unknown dist type: {spec['type']!r}")


class DistStream(IterableDataset):
    '''Infinite stream of fresh distribution-conditioned samples drawn from `DistPriors`.

    Wrap with `DataLoader(stream, batch_size=B, num_workers=...)` to batch on the fly.
    Each yielded item is `{type, params, samples}` — same shape as `DistDataset`.

    Multi-worker safe: each worker's RNG is offset by `worker_id` so they don't
    produce identical streams.
    '''

    def __init__(self, priors, N=128, seed=None):
        self.priors = priors
        self.N = N
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or self.seed is None:
            base_seed = self.seed
        else:
            base_seed = self.seed + worker_info.id
        gen = np.random.default_rng(base_seed)
        while True:
            spec = sample_spec(self.priors, gen)
            samples = _samples_from_spec(spec, self.N, gen)
            yield {
                "type": spec["type"],
                "params": spec["params"],
                "samples": samples,
            }


class DistDataset(Dataset):
    '''Fixed-config dataset. One item per spec in `specs`. Use for held-out validation
    sets where you want the same configs every epoch.

    Each `__getitem__` returns `{type, params, samples}` so the downstream tokenizer
    can assemble `[DIST=...][params...][START] s1 ... sN`.

    Seeding: if `seed` is set, item `idx` uses `seed + idx` so __getitem__ is
    reproducible per index. If `seed=None`, fresh samples on every call.
    '''

    def __init__(self, specs, N=128, seed=None):
        self.specs = list(specs)
        self.N = N
        self.seed = seed

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        item_seed = None if self.seed is None else self.seed + idx
        dist_type = spec["type"]
        params = spec.get("params", {})

        if dist_type == "UNIFORM":
            samples = generate_discrete_uniform(N=self.N, seed=item_seed)
        elif dist_type == "GAUSSIAN":
            samples = generate_discrete_gaussian(N=self.N, seed=item_seed, **params)
        elif dist_type == "BIMODAL":
            samples = generate_discrete_bimodal_gaussian(N=self.N, seed=item_seed, **params)
        else:
            raise ValueError(f"unknown dist type: {dist_type!r}")

        return {
            "type": dist_type,
            "params": params,
            "samples": samples,
        }


@dataclass
class BinnedDistTokenizer:
    '''Tokenizer for conditional distribution sampling with binned float values.

    Fixed prefix:
    `[TYPE] [MU1] [SIGMA1] [MU2] [SIGMA2] [W] [BOS] sample...`
    Missing params use `PARAM_NULL`.
    '''
    value_range: tuple = (0.0, 100.0)
    sigma_range: tuple = (1.0, 15.0)
    w_range: tuple = (0.0, 1.0)
    num_value_bins: int = 128
    num_param_bins: int = 64
    num_w_bins: int = 64

    pad_token: int = 0
    bos_token: int = 1
    uniform_token: int = 2
    gaussian_token: int = 3
    bimodal_token: int = 4
    param_null_token: int = 5

    @property
    def sample_offset(self):
        return 6

    @property
    def mu_offset(self):
        return self.sample_offset + self.num_value_bins

    @property
    def sigma_offset(self):
        return self.mu_offset + self.num_param_bins

    @property
    def w_offset(self):
        return self.sigma_offset + self.num_param_bins

    @property
    def vocab_size(self):
        return self.w_offset + self.num_w_bins

    @property
    def prefix_len(self):
        return 7

    def _bin(self, value, value_range, num_bins):
        lo, hi = value_range
        value = np.clip(value, lo, hi)
        scaled = (value - lo) / (hi - lo)
        return np.minimum((scaled * num_bins).astype(np.int64), num_bins - 1)

    def _mu_token(self, value):
        return int(self.mu_offset + self._bin(value, self.value_range, self.num_param_bins))

    def _sigma_token(self, value):
        return int(self.sigma_offset + self._bin(value, self.sigma_range, self.num_param_bins))

    def _w_token(self, value):
        return int(self.w_offset + self._bin(value, self.w_range, self.num_w_bins))

    def type_token(self, dist_type):
        if dist_type == "UNIFORM":
            return self.uniform_token
        if dist_type == "GAUSSIAN":
            return self.gaussian_token
        if dist_type == "BIMODAL":
            return self.bimodal_token
        raise ValueError(f"unknown dist type: {dist_type!r}")

    def encode_spec(self, spec):
        params = spec.get("params", {})
        null = self.param_null_token
        if spec["type"] == "UNIFORM":
            return [self.uniform_token, null, null, null, null, null, self.bos_token]
        if spec["type"] == "GAUSSIAN":
            return [
                self.gaussian_token,
                self._mu_token(params["mu"]),
                self._sigma_token(params["sigma"]),
                null,
                null,
                null,
                self.bos_token,
            ]
        if spec["type"] == "BIMODAL":
            return [
                self.bimodal_token,
                self._mu_token(params["mu1"]),
                self._sigma_token(params["sigma1"]),
                self._mu_token(params["mu2"]),
                self._sigma_token(params["sigma2"]),
                self._w_token(params["w"]),
                self.bos_token,
            ]
        raise ValueError(f"unknown dist type: {spec['type']!r}")

    def encode_samples(self, samples):
        bins = self._bin(np.asarray(samples), self.value_range, self.num_value_bins)
        return (self.sample_offset + bins).astype(np.int64)

    def encode_item(self, item):
        prefix = np.asarray(self.encode_spec(item), dtype=np.int64)
        samples = self.encode_samples(item["samples"])
        return np.concatenate([prefix, samples])

    def spec_key(self, spec):
        return tuple(self.encode_spec(spec)[:-1])


def _truncated_normal(mu, sigma, N, support, gen):
    if N == 0:
        return np.empty(0, dtype=float)

    lo, hi = support
    samples = []
    needed = N
    while needed > 0:
        draw = gen.normal(mu, sigma, size=max(needed * 2, 32))
        draw = draw[(draw >= lo) & (draw <= hi)]
        samples.append(draw[:needed])
        needed -= len(samples[-1])
    return np.concatenate(samples)


def _continuous_samples_from_spec(spec, N, gen, support=(0.0, 100.0)):
    dist_type = spec["type"]
    params = spec.get("params", {})
    if dist_type == "UNIFORM":
        return gen.uniform(support[0], support[1], size=N)
    if dist_type == "GAUSSIAN":
        return _truncated_normal(params["mu"], params["sigma"], N, support, gen)
    if dist_type == "BIMODAL":
        component_1 = gen.random(N) < params["w"]
        samples = np.empty(N, dtype=float)
        n1 = int(component_1.sum())
        samples[component_1] = _truncated_normal(params["mu1"], params["sigma1"], n1, support, gen)
        samples[~component_1] = _truncated_normal(params["mu2"], params["sigma2"], N - n1, support, gen)
        return samples
    raise ValueError(f"unknown dist type: {dist_type!r}")


def make_eval_specs(priors, n_specs, seed, tokenizer):
    gen = np.random.default_rng(seed)
    specs = []
    keys = set()
    while len(specs) < n_specs:
        spec = sample_spec(priors, gen)
        key = tokenizer.spec_key(spec)
        if key in keys:
            continue
        specs.append(spec)
        keys.add(key)
    return specs


def batch_from_items(items, tokenizer):
    sequences = [tokenizer.encode_item(item) for item in items]
    batch = np.stack(sequences)
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    positions = np.arange(labels.shape[1])
    loss_mask = positions[None, :] >= (tokenizer.prefix_len - 1)
    return inputs, labels, loss_mask.astype(np.float32)


class BinnedDistStream(IterableDataset):
    '''Infinite stream of continuous samples plus binned model tokens.'''

    def __init__(self, priors, tokenizer, N=128, seed=None, excluded_keys=None):
        self.priors = priors
        self.tokenizer = tokenizer
        self.N = N
        self.seed = seed
        self.excluded_keys = set() if excluded_keys is None else set(excluded_keys)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or self.seed is None:
            base_seed = self.seed
        else:
            base_seed = self.seed + worker_info.id
        gen = np.random.default_rng(base_seed)
        while True:
            spec = sample_spec(self.priors, gen)
            if self.tokenizer.spec_key(spec) in self.excluded_keys:
                continue
            samples = _continuous_samples_from_spec(spec, self.N, gen, self.tokenizer.value_range)
            yield {
                "type": spec["type"],
                "params": spec["params"],
                "samples": samples,
            }


class BinnedDistDataset(Dataset):
    '''Fixed held-out specs with fresh continuous samples per spec.'''

    def __init__(self, specs, tokenizer, N=128, seed=None):
        self.specs = list(specs)
        self.tokenizer = tokenizer
        self.N = N
        self.seed = seed

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        gen = np.random.default_rng(None if self.seed is None else self.seed + idx)
        spec = self.specs[idx]
        samples = _continuous_samples_from_spec(spec, self.N, gen, self.tokenizer.value_range)
        return {
            "type": spec["type"],
            "params": spec.get("params", {}),
            "samples": samples,
        }
