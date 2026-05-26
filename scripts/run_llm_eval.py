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
import tempfile
from pathlib import Path


SUMMARY_KEYS = ('mu_requested', 'sigma_requested', 'mu_observed',
                'std_observed', 'parse_rate', 'malformed', 'n_clean',
                'ks_stat', 'ks_pvalue')


def _atomic_write_bytes(path: Path, data: bytes):
    """Write `data` to `path` via a tmp file + rename on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _save(out_path: Path, args, results, *, fallback_dir: Path | None = None):
    """Save pickle + json sidecar. On OSError, retry under fallback_dir."""
    payload = pickle.dumps({'args': vars(args), 'results': results})
    summary = [{k: r[k] for k in SUMMARY_KEYS} for r in results]
    summary_bytes = json.dumps(
        {'args': vars(args), 'summary': summary}, indent=2,
    ).encode()
    summary_path = out_path.with_suffix(out_path.suffix + '.json')

    try:
        _atomic_write_bytes(out_path, payload)
        _atomic_write_bytes(summary_path, summary_bytes)
        return out_path, summary_path
    except OSError as e:
        if fallback_dir is None:
            raise
        fb_out = fallback_dir / out_path.name
        fb_summary = fallback_dir / summary_path.name
        print(f"!! save to {out_path} failed ({e}); falling back to {fb_out}")
        _atomic_write_bytes(fb_out, payload)
        _atomic_write_bytes(fb_summary, summary_bytes)
        return fb_out, fb_summary


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
    ap.add_argument('--device-map', default=None,
                    help="passed to from_pretrained (e.g. 'auto'). Default: "
                         "load on CPU then .to(--device) to avoid accelerate "
                         "offload hooks for small models.")
    ap.add_argument('--device', default='cuda',
                    help="device to move the model to when --device-map is "
                         "unset. Use 'cuda', 'cuda:0', or 'cpu'.")
    ap.add_argument('--batch-size', type=int, default=0,
                    help="num_return_sequences per generate call. 0 = run all "
                         "n_runs in one batched call per mu (recommended on GPU).")
    args = ap.parse_args()

    cache_dir = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ.setdefault('HF_HUB_CACHE', f"{cache_dir}/hub")

    out_path = Path(args.out).expanduser().resolve()
    fallback_dir = (cache_dir / 'llm_eval_runs') if cache_dir else None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"!! cannot create {out_path.parent} ({e}); will use fallback")

    # Imports after env vars are set so HF picks them up.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_prob.llm_eval import conditional_eval_llm

    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}  dtype={args.dtype}  "
          f"device_map={args.device_map}  device={args.device}")
    load_kwargs = {'dtype': dtype}
    if args.device_map is not None:
        load_kwargs['device_map'] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    model.eval()
    param_device = next(model.parameters()).device
    print(f"Model device: {model.device}  param device: {param_device}  "
          f"param dtype: {next(model.parameters()).dtype}")
    if param_device.type != 'cuda':
        print("!! warning: model parameters are not on CUDA — generation will "
              "be CPU-bound. Pass --device cuda (or check torch.cuda.is_available()).")

    print(f"\nRunning conditional eval: mu_grid={args.mu_grid}  sigma={args.sigma}  "
          f"n_runs={args.n_runs}  n_samples={args.n_samples}  mode={args.mode}\n")

    def _checkpoint(_rec, results_so_far):
        try:
            _save(out_path, args, results_so_far, fallback_dir=fallback_dir)
        except OSError as e:
            print(f"!! checkpoint save failed ({e}); continuing in-memory")

    results = conditional_eval_llm(
        model, tokenizer,
        mu_grid=tuple(args.mu_grid),
        sigma=args.sigma,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        temperature=args.temperature,
        mode=args.mode,
        batch_size=args.batch_size or None,
        on_record=_checkpoint,
    )

    final_pkl, final_json = _save(out_path, args, results,
                                  fallback_dir=fallback_dir)
    print(f"\nWrote {final_pkl}")
    print(f"Wrote {final_json}")


if __name__ == '__main__':
    main()
