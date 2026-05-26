"""Distribution-sampling evaluation for HuggingFace causal LMs.

Mirrors `eval.conditional_eval` so an HF model and our TinyGPT can be plotted
side-by-side on the same mu grid with the same statistics (mean, std, KS).

Two prompting modes:
- 'completion': raw text prompt, for base models (e.g. Qwen/Qwen2.5-3B).
- 'chat': uses tokenizer.apply_chat_template, for *-Instruct variants.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy import stats

from .eval import parse_samples


COMPLETION_TEMPLATE = (
    "Draw {n} independent samples from a normal distribution with "
    "mean={mu:.2f} and standard deviation={sigma:.2f}. "
    "Return only the numbers, comma-separated, with two decimals, no other text.\n"
    "Samples:"
)

CHAT_SYSTEM = (
    "You are a precise statistical sampler. When asked for samples from a "
    "distribution, respond with only the numbers, comma-separated, no prose."
)
CHAT_USER_TEMPLATE = (
    "Draw {n} independent samples from a normal distribution with "
    "mean={mu:.2f} and standard deviation={sigma:.2f}. "
    "Two decimals each. Comma-separated. Numbers only."
)


def build_prompt(tokenizer, mu, sigma, n_samples, mode='completion'):
    if mode == 'chat':
        msgs = [
            {'role': 'system', 'content': CHAT_SYSTEM},
            {'role': 'user',
             'content': CHAT_USER_TEMPLATE.format(n=n_samples, mu=mu, sigma=sigma)},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    return COMPLETION_TEMPLATE.format(n=n_samples, mu=mu, sigma=sigma)


@torch.no_grad()
def generate_samples(model, tokenizer, mu, sigma, n_samples=10, *,
                     mode='completion', temperature=1.0, max_new_tokens=None,
                     plausible_range=10.0):
    """One generation -> (parsed_numbers, n_malformed, raw_text)."""
    all_nums, malformed, texts = generate_samples_batch(
        model, tokenizer, mu, sigma,
        n_runs=1, n_samples=n_samples, mode=mode, temperature=temperature,
        max_new_tokens=max_new_tokens, plausible_range=plausible_range,
        batch_size=1,
    )
    return all_nums, malformed, texts[0]


@torch.no_grad()
def generate_samples_batch(model, tokenizer, mu, sigma, *,
                           n_runs, n_samples=10, mode='completion',
                           temperature=1.0, max_new_tokens=None,
                           plausible_range=10.0, batch_size=None):
    """`n_runs` independent samplings of the same prompt, batched.

    Uses `num_return_sequences` so a single forward pass yields N independent
    sampled continuations — orders of magnitude faster than calling generate
    N times at batch=1 on a GPU.

    Returns (parsed_nums_concat, malformed_total, raw_texts_list).
    """
    prompt = build_prompt(tokenizer, mu, sigma, n_samples, mode=mode)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    if max_new_tokens is None:
        max_new_tokens = n_samples * 10 + 16
    if batch_size is None or batch_size <= 0:
        batch_size = n_runs

    all_nums, malformed_total, raw_texts = [], 0, []
    remaining = n_runs
    while remaining > 0:
        bs = min(batch_size, remaining)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=bs,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for text in texts:
            text_for_parse = text.split('\n')[0]
            nums, bad = parse_samples(text_for_parse,
                                       plausible_range=plausible_range)
            all_nums.extend(nums)
            malformed_total += bad
            raw_texts.append(text)
        remaining -= bs
    return all_nums, malformed_total, raw_texts


@torch.no_grad()
def conditional_eval_llm(model, tokenizer, *,
                         mu_grid=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
                         sigma=1.0, n_runs=50, n_samples=10,
                         temperature=1.0, mode='completion', verbose=True,
                         on_record=None, batch_size=None):
    """Same record shape as eval.conditional_eval — plug into the same plots.

    `on_record(rec, results_so_far)` is invoked after each mu so callers can
    checkpoint incrementally. `batch_size` caps `num_return_sequences` per
    generate call (defaults to n_runs — i.e. one batched call per mu).
    """
    results = []
    cuda_avail = torch.cuda.is_available()
    for mu in mu_grid:
        if cuda_avail:
            torch.cuda.synchronize()
        t_gen = time.perf_counter()
        all_nums, malformed, raw_texts = generate_samples_batch(
            model, tokenizer, mu, sigma,
            n_runs=n_runs, n_samples=n_samples, mode=mode,
            temperature=temperature, batch_size=batch_size,
        )
        if cuda_avail:
            torch.cuda.synchronize()
        gen_dt = time.perf_counter() - t_gen
        all_nums = np.array(all_nums, dtype=float)
        clean = all_nums[np.abs(all_nums - mu) < 5.0]
        expected = n_runs * n_samples
        ks = stats.kstest(clean, 'norm', args=(mu, sigma)) if len(clean) > 30 else None
        rec = {
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
        }
        results.append(rec)
        if verbose:
            print(f"  mu_req={rec['mu_requested']:+.2f}  ->  "
                  f"mu_obs={rec['mu_observed']:+.3f}  "
                  f"sigma_obs={rec['std_observed']:.3f}  "
                  f"parse={rec['parse_rate']:.1%}  bad={rec['malformed']}  "
                  f"KS={rec['ks_stat']:.3f}  n={rec['n_clean']}  "
                  f"[gen+decode {gen_dt:.2f}s]")
        if on_record is not None:
            on_record(rec, results)
    return results
