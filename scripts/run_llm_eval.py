"""First-run script: load a HF causal LM and evaluate it against N(mu, sigma).

Designed for a cluster interactive session. Reads the model from a local
snapshot (downloaded via scripts/download_model.py) so we don't go through
~/.cache.

Usage:
    HF_HOME=/scratch/$USER/hf-cache uv run python scripts/run_llm_eval.py \
        --model Qwen/Qwen2.5-3B \
        --cache-dir /scratch/$USER/hf-cache \
        --out runs/qwen25_3b_first.pkl \
        --mode completion \
        --n-runs 20 --n-samples 10

Outputs:
    - <out>           pickled list of per-mu records (same shape as
                      eval.conditional_eval)
    - <out>.json      a small JSON summary (mu_req, mu_obs, std, KS, parse)
"""

import argparse
import json
import os
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-3B')
    ap.add_argument('--cache-dir', default=None,
                    help='HF cache root. If given, also sets HF_HOME so the '
                         'tokenizer/model load from there.')
    ap.add_argument('--out', default='runs/llm_eval.pkl')
    ap.add_argument('--mode', choices=['completion', 'chat'], default='completion',
                    help='completion for base models, chat for *-Instruct')
    ap.add_argument('--mu-grid', type=float, nargs='+',
                    default=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--n-runs', type=int, default=20)
    ap.add_argument('--n-samples', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    ap.add_argument('--device-map', default='auto',
                    help="passed to from_pretrained; 'auto' uses accelerate")
    args = ap.parse_args()

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).expanduser().resolve())
        os.environ['HF_HOME'] = cache_dir
        os.environ.setdefault('HF_HUB_CACHE', f"{cache_dir}/hub")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Imports after env vars are set so HF picks them up.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_prob.llm_eval import conditional_eval_llm

    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}  dtype={args.dtype}  device_map={args.device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()
    print(f"Model device: {model.device}  param dtype: {next(model.parameters()).dtype}")

    print(f"\nRunning conditional eval: mu_grid={args.mu_grid}  sigma={args.sigma}  "
          f"n_runs={args.n_runs}  n_samples={args.n_samples}  mode={args.mode}\n")
    results = conditional_eval_llm(
        model, tokenizer,
        mu_grid=tuple(args.mu_grid),
        sigma=args.sigma,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        temperature=args.temperature,
        mode=args.mode,
    )

    with open(out_path, 'wb') as f:
        pickle.dump({'args': vars(args), 'results': results}, f)
    print(f"\nWrote {out_path}")

    summary = [{k: r[k] for k in
                ('mu_requested', 'sigma_requested', 'mu_observed',
                 'std_observed', 'parse_rate', 'malformed', 'n_clean',
                 'ks_stat', 'ks_pvalue')}
               for r in results]
    summary_path = out_path.with_suffix(out_path.suffix + '.json')
    with open(summary_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary}, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == '__main__':
    main()
