"""Diagnostic plots for distribution-modelling evals."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .data import collate


def plot_eval_distribution(eval_history, step):
    if not eval_history:
        raise ValueError('eval_history is empty')

    record = min(eval_history, key=lambda r: abs(r['step'] - step))
    by_mean = record['by_mean']
    n = len(by_mean)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)

    clean_by_mean = {}
    for row, (mu, rec) in enumerate(sorted(by_mean.items())):
        sigma = rec['sigma']
        all_nums = rec['all_nums']
        clean = all_nums[np.abs(all_nums) < 10]
        clean_by_mean[mu] = clean

        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        axes[row, 0].hist(clean, bins=50, density=True, alpha=0.6, label=f"model (n={len(clean)})")
        axes[row, 0].plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', lw=2, label=f'true N({mu:.1f},{sigma:.1f})')
        axes[row, 0].set_title(f"Density at mu={mu:+.1f}, step {record['step']}")
        axes[row, 0].legend()

        stats.probplot(clean, dist=stats.norm(loc=mu, scale=sigma), plot=axes[row, 1])
        axes[row, 1].set_title(f'Q-Q plot at mu={mu:+.1f}')

    plt.tight_layout()
    plt.show()
    return clean_by_mean, record


def plot_entropy_by_position(model, eval_examples, n_eval=256, device='cpu'):
    eval_toks, eval_mask = collate(eval_examples[:n_eval])
    eval_toks, eval_mask = eval_toks.to(device), eval_mask.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(eval_toks[:, :-1])
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum(-1)
        mask = eval_mask[:, 1:].bool()
        per_pos = (ent * mask).sum(0) / mask.sum(0).clamp(min=1)
    model.train()

    plt.plot(per_pos.cpu().numpy())
    plt.xlabel('position')
    plt.ylabel('mean entropy (nats)')
    plt.title('Uncertainty by position')
    plt.show()
    return per_pos.cpu().numpy()


def plot_conditional_eval(results, sigma=1.0):
    mus = np.array([r['mu_requested'] for r in results], dtype=float)
    obs = np.array([r['mu_observed'] for r in results], dtype=float)
    stds = np.array([r['std_observed'] for r in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(mus, obs, 'o-', label='model')
    axes[0].plot(mus, mus, 'k--', label='ideal')
    axes[0].set_xlabel('requested mu')
    axes[0].set_ylabel('observed mean')
    axes[0].set_title('Conditional mean tracking')
    axes[0].legend()

    axes[1].plot(mus, stds, 'o-', label='model')
    axes[1].axhline(sigma, color='k', linestyle='--', label='ideal sigma')
    axes[1].set_xlabel('requested mu')
    axes[1].set_ylabel('observed std')
    axes[1].set_title('Conditional variance stability')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(len(results), 1, figsize=(7, 2.4 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        mu = r['mu_requested']
        clean = r.get('clean')
        if clean is None:
            samples = r.get('samples', np.array([]))
            clean = samples[np.abs(samples - mu) < 5.0]
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        ax.hist(clean, bins=40, density=True, alpha=0.6, label=f"model (n={len(clean)})")
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', lw=2, label='truth')
        ax.set_title(f"mu={mu:+.2f}: generated density vs N(mu, {sigma:g})")
        ax.legend()

    plt.tight_layout()
    plt.show()
    return results
