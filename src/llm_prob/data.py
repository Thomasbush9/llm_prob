"""Dataset builders and a collate fn that emits per-token loss masks.

Each example is `(toks: list[int], sample_start: int)`. `sample_start` is the
index just after `<SEP>` — the loss mask is 1 from there to the end of the
real sequence and 0 elsewhere (padding excluded).
"""

import numpy as np
import torch

from .tokenization import PAD, build_sequence


def collate(batch):
    max_len = max(len(b[0]) for b in batch)
    toks = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (seq, start) in enumerate(batch):
        seq_t = torch.tensor(seq, dtype=torch.long)
        toks[i, :len(seq_t)] = seq_t
        mask[i, start:len(seq_t)] = 1
    return toks, mask


def make_fixed_dataset(n_examples, n_samples=10, mu=0.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n_examples):
        samples = rng.normal(mu, sigma, n_samples)
        toks, start = build_sequence('normal', [mu, sigma], samples)
        examples.append((toks, start))
    return examples


def make_mean_sweep_dataset(n_examples, n_samples=10,
                            means=(-2.0, -1.0, 0.0, 1.0, 2.0),
                            sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    means = np.asarray(means, dtype=float)
    examples = []
    for _ in range(n_examples):
        mu = float(rng.choice(means))
        samples = rng.normal(mu, sigma, n_samples)
        toks, start = build_sequence('normal', [mu, sigma], samples)
        examples.append((toks, start))
    return examples


def make_continuous_mu_dataset(n_examples, n_samples=10,
                               mu_range=(-3.0, 3.0), sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    examples, metadata = [], []
    for _ in range(n_examples):
        mu = float(rng.uniform(*mu_range))
        samples = rng.normal(mu, sigma, n_samples)
        toks, start = build_sequence('normal', [mu, sigma], samples)
        examples.append((toks, start))
        metadata.append({'mu': mu, 'sigma': sigma})
    return examples, metadata
