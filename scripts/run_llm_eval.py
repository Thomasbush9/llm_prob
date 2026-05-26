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
import functools
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path


# Line-buffered output so cluster logs / `tee` show progress in real time
# instead of dumping everything after the run finishes.
print = functools.partial(print, flush=True)  # noqa: A001


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
    ap.add_argument('--checkpoint-every', type=int, default=0,
                    help="checkpoint to disk every N mus (0 = only at end). "
                         "Per-mu checkpointing on slow shared filesystems "
                         "stalls the GPU; leave at 0 unless you need it.")
    ap.add_argument('--no-warmup', action='store_true',
                    help="skip the warmup generate() call before the main loop")
    ap.add_argument('--smoke-test', action='store_true',
                    help="load model, run one tiny generate, report timing, exit. "
                         "Use to isolate GPU perf from the full eval.")
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
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_prob.llm_eval import conditional_eval_llm, generate_samples_batch

    print(f"python={sys.version.split()[0]}  torch={torch.__version__}  "
          f"cuda={torch.version.cuda}  transformers={transformers.__version__}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}  "
          f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}  "
          f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")

    want_cuda = args.device.startswith('cuda') and args.device_map is None
    if want_cuda and not torch.cuda.is_available():
        sys.exit(f"!! requested --device {args.device} but torch.cuda.is_available() "
                 "is False. Either allocate a GPU (e.g. salloc/srun with --gres=gpu:1) "
                 "or pass --device cpu. Refusing to silently fall back to CPU.")

    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]

    print(f"Loading tokenizer: {args.model}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  tokenizer loaded in {time.perf_counter() - t0:.2f}s")

    print(f"Loading model: {args.model}  dtype={args.dtype}  "
          f"device_map={args.device_map}  device={args.device}")
    t0 = time.perf_counter()
    load_kwargs = {'dtype': dtype}
    if args.device_map is not None:
        load_kwargs['device_map'] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    model.eval()
    print(f"  model loaded + on device in {time.perf_counter() - t0:.2f}s")
    param_device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    print(f"Model device: {model.device}  param device: {param_device}  "
          f"param dtype: {param_dtype}")
    if param_dtype != dtype:
        print(f"!! warning: requested dtype={dtype} but params are {param_dtype}. "
              "The dtype kwarg may not have been respected by from_pretrained.")
    if param_device.type != 'cuda':
        print("!! warning: model parameters are not on CUDA — generation will "
              "be CPU-bound.")

    if not args.no_warmup and param_device.type == 'cuda':
        print("Warmup generate() (kernel autotune)...")
        t0 = time.perf_counter()
        warm = tokenizer("Hello", return_tensors='pt').to(param_device)
        _ = model.generate(**warm, max_new_tokens=8, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        print(f"  warmup done in {time.perf_counter() - t0:.2f}s")

    if args.smoke_test:
        print("\n=== smoke test ===")
        print("Single batched generate call: mu=0, n_runs=n_runs, n_samples=n_samples")
        if param_device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        nums, bad, texts = generate_samples_batch(
            model, tokenizer, mu=0.0, sigma=args.sigma,
            n_runs=args.n_runs, n_samples=args.n_samples, mode=args.mode,
            temperature=args.temperature,
            batch_size=args.batch_size or None,
        )
        if param_device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  generate+decode: {dt:.2f}s for n_runs={args.n_runs} "
              f"(batch_size={args.batch_size or args.n_runs}, "
              f"max_new_tokens={args.n_samples * 10 + 16})")
        print(f"  parsed {len(nums)} numbers, {bad} malformed")
        print(f"  sample text[0]: {texts[0][:200]!r}")
        return

    print(f"\nRunning conditional eval: mu_grid={args.mu_grid}  sigma={args.sigma}  "
          f"n_runs={args.n_runs}  n_samples={args.n_samples}  mode={args.mode}  "
          f"checkpoint_every={args.checkpoint_every}\n")

    mu_counter = {'n': 0}

    def _checkpoint(_rec, results_so_far):
        mu_counter['n'] += 1
        if args.checkpoint_every <= 0:
            return
        if mu_counter['n'] % args.checkpoint_every != 0:
            return
        t0 = time.perf_counter()
        try:
            _save(out_path, args, results_so_far, fallback_dir=fallback_dir)
        except OSError as e:
            print(f"!! checkpoint save failed ({e}); continuing in-memory")
            return
        print(f"  [checkpoint after mu#{mu_counter['n']} in "
              f"{time.perf_counter() - t0:.2f}s]")

    t_eval = time.perf_counter()
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
    print(f"\nEval loop wall time: {time.perf_counter() - t_eval:.2f}s")

    t0 = time.perf_counter()
    final_pkl, final_json = _save(out_path, args, results,
                                  fallback_dir=fallback_dir)
    print(f"Final save in {time.perf_counter() - t0:.2f}s")
    print(f"Wrote {final_pkl}")
    print(f"Wrote {final_json}")


if __name__ == '__main__':
    main()
