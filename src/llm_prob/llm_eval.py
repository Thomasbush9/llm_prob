"""Distribution-sampling evaluation for HuggingFace causal LMs.

Mirrors `eval.conditional_eval` so an HF model and our TinyGPT can be plotted
side-by-side on the same mu grid with the same statistics (mean, std, KS).

Two prompting modes:
- 'completion': raw text prompt, for base models (e.g. Qwen/Qwen2.5-3B).
- 'chat': uses tokenizer.apply_chat_template, for *-Instruct variants.
"""

from __future__ import annotations

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
    prompt = build_prompt(tokenizer, mu, sigma, n_samples, mode=mode)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    if max_new_tokens is None:
        max_new_tokens = n_samples * 10 + 16

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Cut at first newline — base models often continue with extra text.
    text_for_parse = text.split('\n')[0]
    nums, malformed = parse_samples(text_for_parse, plausible_range=plausible_range)
    return nums, malformed, text


@torch.no_grad()
def conditional_eval_llm(model, tokenizer, *,
                         mu_grid=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
                         sigma=1.0, n_runs=50, n_samples=10,
                         temperature=1.0, mode='completion', verbose=True,
                         on_record=None):
    """Same record shape as eval.conditional_eval — plug into the same plots.

    `on_record(rec, results_so_far)` is invoked after each mu so callers can
    checkpoint incrementally.
    """
    results = []
    for mu in mu_grid:
        all_nums, malformed, raw_texts = [], 0, []
        for _ in range(n_runs):
            nums, bad, text = generate_samples(
                model, tokenizer, mu, sigma,
                n_samples=n_samples, mode=mode, temperature=temperature,
            )
            all_nums.extend(nums)
            malformed += bad
            raw_texts.append(text)

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
                  f"KS={rec['ks_stat']:.3f}  n={rec['n_clean']}")
        if on_record is not None:
            on_record(rec, results)
    return results
