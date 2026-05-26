from dataclasses import dataclass
from math import erf, sqrt

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
    num_value_bins: int = 512
    num_param_bins: int = 128
    num_w_bins: int = 128

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

    def decode_samples(self, tokens):
        bins = np.asarray(tokens, dtype=np.int64) - self.sample_offset
        valid = (0 <= bins) & (bins < self.num_value_bins)
        centers = self.bin_centers()
        decoded = np.full(bins.shape, np.nan, dtype=float)
        decoded[valid] = centers[bins[valid]]
        return decoded, valid

    def bin_edges(self):
        return np.linspace(self.value_range[0], self.value_range[1], self.num_value_bins + 1)

    def bin_centers(self):
        edges = self.bin_edges()
        return 0.5 * (edges[:-1] + edges[1:])

    def encode_item(self, item):
        prefix = np.asarray(self.encode_spec(item), dtype=np.int64)
        samples = self.encode_samples(item["samples"])
        return np.concatenate([prefix, samples])

    def spec_key(self, spec):
        return tuple(self.encode_spec(spec)[:-1])


@dataclass
class DigitDistTokenizer:
    '''LLM-like tokenizer that writes distribution params and samples as digits.

    Each numeric field has fixed width, so batches can be stacked without padding:
    `[TYPE] [MU1] +050.00 [SIGMA1] +010.00 ... [SAMPLES] +049.21 [SEP] ...`.
    '''
    value_range: tuple = (0.0, 100.0)
    sigma_range: tuple = (1.0, 15.0)
    w_range: tuple = (0.0, 1.0)
    num_value_bins: int = 512
    num_param_bins: int = 128
    num_w_bins: int = 128
    number_width: int = 7
    decimals: int = 2

    pad_token: int = 0
    bos_token: int = 1
    sep_token: int = 2
    samples_token: int = 3
    param_null_token: int = 4
    uniform_token: int = 5
    gaussian_token: int = 6
    bimodal_token: int = 7
    mu1_token: int = 8
    sigma1_token: int = 9
    mu2_token: int = 10
    sigma2_token: int = 11
    w_token: int = 12

    @property
    def char_tokens(self):
        return {
            "+": 13,
            "-": 14,
            ".": 15,
            "0": 16,
            "1": 17,
            "2": 18,
            "3": 19,
            "4": 20,
            "5": 21,
            "6": 22,
            "7": 23,
            "8": 24,
            "9": 25,
        }

    @property
    def id_to_char(self):
        return {token: char for char, token in self.char_tokens.items()}

    @property
    def vocab_size(self):
        return 26

    @property
    def prefix_len(self):
        return 1 + 5 * (1 + self.number_width) + 1

    @property
    def sample_token_width(self):
        return self.number_width + 1

    def _format_number(self, value):
        value = float(np.clip(value, *self.value_range))
        text = f"{value:+0{self.number_width}.{self.decimals}f}"
        if len(text) != self.number_width:
            raise ValueError(f"formatted value {text!r} does not match width {self.number_width}")
        return text

    def _encode_number(self, value):
        return [self.char_tokens[char] for char in self._format_number(value)]

    def _encode_null_number(self):
        return [self.param_null_token] * self.number_width

    def _field(self, name_token, value):
        number = self._encode_null_number() if value is None else self._encode_number(value)
        return [name_token] + number

    def encode_spec(self, spec):
        params = spec.get("params", {})
        if spec["type"] == "UNIFORM":
            values = (None, None, None, None, None)
            type_token = self.uniform_token
        elif spec["type"] == "GAUSSIAN":
            values = (params["mu"], params["sigma"], None, None, None)
            type_token = self.gaussian_token
        elif spec["type"] == "BIMODAL":
            values = (params["mu1"], params["sigma1"], params["mu2"], params["sigma2"], params["w"])
            type_token = self.bimodal_token
        else:
            raise ValueError(f"unknown dist type: {spec['type']!r}")

        tokens = [type_token]
        for name_token, value in zip(
            (self.mu1_token, self.sigma1_token, self.mu2_token, self.sigma2_token, self.w_token),
            values,
        ):
            tokens.extend(self._field(name_token, value))
        tokens.append(self.samples_token)
        return tokens

    def encode_samples(self, samples):
        tokens = []
        for sample in samples:
            tokens.extend(self._encode_number(sample))
            tokens.append(self.sep_token)
        return np.asarray(tokens, dtype=np.int64)

    def encode_item(self, item):
        prefix = np.asarray(self.encode_spec(item), dtype=np.int64)
        samples = self.encode_samples(item["samples"])
        return np.concatenate([prefix, samples])

    def decode_samples(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)
        values = []
        valid = []
        for start in range(0, len(tokens), self.sample_token_width):
            chunk = tokens[start:start + self.sample_token_width]
            if len(chunk) < self.sample_token_width:
                break
            number_tokens = chunk[:self.number_width]
            sep = chunk[-1]
            text = "".join(self.id_to_char.get(int(token), "?") for token in number_tokens)
            try:
                value = float(text)
                ok = sep == self.sep_token and self.value_range[0] <= value <= self.value_range[1]
            except ValueError:
                value = np.nan
                ok = False
            values.append(value if ok else np.nan)
            valid.append(ok)
        return np.asarray(values, dtype=float), np.asarray(valid, dtype=bool)

    def bin_edges(self):
        return np.linspace(self.value_range[0], self.value_range[1], self.num_value_bins + 1)

    def bin_centers(self):
        edges = self.bin_edges()
        return 0.5 * (edges[:-1] + edges[1:])

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


def _normal_cdf(x, mu, sigma):
    z = (np.asarray(x, dtype=float) - mu) / (sigma * sqrt(2.0))
    return np.vectorize(erf)(z) * 0.5 + 0.5


def _truncated_normal_bin_probs(mu, sigma, edges):
    lo, hi = edges[0], edges[-1]
    normalizer = _normal_cdf(hi, mu, sigma) - _normal_cdf(lo, mu, sigma)
    probs = np.diff(_normal_cdf(edges, mu, sigma)) / normalizer
    return probs / probs.sum()


def true_binned_probs(spec, tokenizer):
    '''Target probability mass over tokenizer value bins for one distribution spec.'''
    edges = tokenizer.bin_edges()
    dist_type = spec["type"]
    params = spec.get("params", {})
    if dist_type == "UNIFORM":
        probs = np.ones(len(edges) - 1, dtype=float)
        return probs / probs.sum()
    if dist_type == "GAUSSIAN":
        return _truncated_normal_bin_probs(params["mu"], params["sigma"], edges)
    if dist_type == "BIMODAL":
        p1 = _truncated_normal_bin_probs(params["mu1"], params["sigma1"], edges)
        p2 = _truncated_normal_bin_probs(params["mu2"], params["sigma2"], edges)
        probs = params["w"] * p1 + (1.0 - params["w"]) * p2
        return probs / probs.sum()
    raise ValueError(f"unknown dist type: {dist_type!r}")


def empirical_binned_probs(values, tokenizer, eps=1e-12):
    valid_values = np.asarray(values, dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]
    if len(valid_values) == 0:
        probs = np.ones(tokenizer.num_value_bins, dtype=float)
        return probs / probs.sum()
    counts, _ = np.histogram(valid_values, bins=tokenizer.bin_edges())
    probs = counts.astype(float) + eps
    return probs / probs.sum()


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
    loss_mask = np.broadcast_to(loss_mask, labels.shape)
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
