"""Sampling, parsing, and aggregate eval against true normal distributions."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .data import collate
from .tokenization import BOS, EOS, ITOS, P, SEP, SPECIALS, STOI, encode_number, fmt


def parse_samples(text, plausible_range=10.0):
    nums, malformed = [], 0
    for s in text.split(','):
        s = s.strip()
        try:
            x = float(s)
            if abs(x) <= plausible_range:
                nums.append(x)
            else:
                malformed += 1
        except ValueError:
            malformed += 1
    return nums, malformed


@torch.no_grad()
def generate(model, dist='normal', params=(0.0, 1.0), n_samples=10,
             max_new_tokens=None, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    toks = [BOS, STOI[f'<{dist.upper()}>']]
    for p in params:
        toks.append(P)
        toks.extend(encode_number(fmt(p)))
    toks.append(SEP)

    if max_new_tokens is None:
        max_new_tokens = n_samples * 8 + 8

    for _ in range(max_new_tokens):
        x = torch.tensor(toks, device=device).unsqueeze(0)
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        toks.append(nxt)
        if nxt == EOS:
            break

    tail_start = toks.index(SEP) + 1
    text = ''.join(ITOS[t] for t in toks[tail_start:] if ITOS[t] not in SPECIALS)
    nums, malformed = parse_samples(text)
    model.train()
    return nums, malformed, text


def _eval_loss(model, eval_examples, device, batch_size=256):
    toks, mask = collate(eval_examples[:batch_size])
    toks, mask = toks.to(device), mask.to(device)
    logits = model(toks[:, :-1])
    targets = toks[:, 1:]
    smask = mask[:, 1:].float()
    lpt = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
    )
    return float(((lpt * smask.reshape(-1)).sum() / smask.sum()).item())


@torch.no_grad()
def run_eval(model, eval_examples, eval_means=(-2.0, -1.0, 0.0, 1.0, 2.0),
             sigma=1.0, n_runs=50, n_samples=10, step=None, device='cpu'):
    model.eval()
    eval_loss = _eval_loss(model, eval_examples, device)

    by_mean = {}
    for mu in eval_means:
        all_nums, malformed, raw_texts = [], 0, []
        for _ in range(n_runs):
            nums, bad, text = generate(model, 'normal', (mu, sigma), n_samples=n_samples)
            all_nums.extend(nums)
            malformed += bad
            raw_texts.append(text)

        all_nums = np.array(all_nums, dtype=float)
        expected = n_runs * n_samples
        rec = {
            'mu': float(mu),
            'sigma': float(sigma),
            'parse_rate': float(len(all_nums) / expected),
            'malformed': int(malformed),
            'all_nums': all_nums,
            'raw_texts': raw_texts,
        }
        if len(all_nums) > 30:
            ks = stats.kstest(all_nums, 'norm', args=(mu, sigma))
            rec.update({
                'n': int(len(all_nums)),
                'mean': float(all_nums.mean()),
                'std': float(all_nums.std()),
                'ks_stat': float(ks.statistic),
                'ks_pvalue': float(ks.pvalue),
            })
        by_mean[float(mu)] = rec

    record = {
        'step': step,
        'eval_loss': eval_loss,
        'eval_means': tuple(float(m) for m in eval_means),
        'sigma': float(sigma),
        'by_mean': by_mean,
    }

    print(f"  step={step}  eval_loss={eval_loss:.4f}")
    for mu, rec in by_mean.items():
        if 'ks_stat' in rec:
            print(f"    mu={mu:+.1f}  parse={rec['parse_rate']:.1%}  bad={rec['malformed']}  "
                  f"mean={rec['mean']:.2f}  std={rec['std']:.2f}  KS={rec['ks_stat']:.3f}")
        else:
            print(f"    mu={mu:+.1f}  parse={rec['parse_rate']:.1%}  bad={rec['malformed']}  too few samples")

    model.train()
    return record


@torch.no_grad()
def conditional_eval(model, mu_grid=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
                     sigma=1.0, n_runs=50, n_samples=10, temperature=1.0):
    results = []
    for mu in mu_grid:
        all_nums, malformed, raw_texts = [], 0, []
        for _ in range(n_runs):
            nums, bad, text = generate(
                model, 'normal', (mu, sigma),
                n_samples=n_samples, temperature=temperature,
            )
            all_nums.extend(nums)
            malformed += bad
            raw_texts.append(text)

        all_nums = np.array(all_nums, dtype=float)
        clean = all_nums[np.abs(all_nums - mu) < 5.0]
        expected = n_runs * n_samples
        ks = stats.kstest(clean, 'norm', args=(mu, sigma)) if len(clean) > 30 else None
        results.append({
            'mu_requested': float(mu),
            'sigma_requested': float(sigma),
            'mu_observed': float(clean.mean()) if len(clean) else float('nan'),
            'std_observed': float(clean.std()) if len(clean) else float('nan'),
            'parse_rate': float(len(all_nums) / expected),
            'malformed': int(malformed),
            'n_clean': int(len(clean)),
            'ks_stat': float(ks.statistic) if ks else float('nan'),
            'ks_pvalue': float(ks.pvalue) if ks else float('nan'),
            'samples': all_nums,
            'clean': clean,
            'raw_texts': raw_texts,
        })

    for r in results:
        print(f"  mu_req={r['mu_requested']:+.2f}  ->  "
              f"mu_obs={r['mu_observed']:+.3f}  "
              f"sigma_obs={r['std_observed']:.3f}  "
              f"parse={r['parse_rate']:.1%}  bad={r['malformed']}  "
              f"KS={r['ks_stat']:.3f}  n={r['n_clean']}")
    return results
