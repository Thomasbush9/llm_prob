"""Snapshot-download a Hugging Face model into a user-chosen cache dir.

The cluster home dir is usually quota-limited, so the default HF cache
(~/.cache/huggingface) is a bad place for multi-GB checkpoints. This script
puts the snapshot wherever you point it.

Usage:
    uv run python scripts/download_model.py \
        --model Qwen/Qwen2.5-3B \
        --cache-dir /scratch/$USER/hf-cache

After this runs, point `transformers` at the same dir via either:
    HF_HOME=/scratch/$USER/hf-cache  (env var; covers cache + tokenizers + hub)
or by passing `cache_dir=...` to `from_pretrained`.
"""

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-3B',
                    help='HF repo id (default: Qwen/Qwen2.5-3B)')
    ap.add_argument('--cache-dir', required=True,
                    help='Local directory to store the snapshot. '
                         'Will be created if missing.')
    ap.add_argument('--revision', default=None, help='Optional git revision/branch/tag')
    ap.add_argument('--token', default=None,
                    help='HF token (or set HUGGING_FACE_HUB_TOKEN env var)')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HOME'] = str(cache_dir)
    os.environ.setdefault('HF_HUB_CACHE', str(cache_dir / 'hub'))

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model} -> {cache_dir}")
    local_path = snapshot_download(
        repo_id=args.model,
        cache_dir=str(cache_dir / 'hub'),
        revision=args.revision,
        token=args.token or os.environ.get('HUGGING_FACE_HUB_TOKEN'),
    )
    print(f"\nSnapshot at: {local_path}")
    print(f"\nTo use it later, run with:")
    print(f"  HF_HOME={cache_dir} uv run python <your script>")


if __name__ == '__main__':
    main()
