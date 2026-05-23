import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx, serialization

from data import (
    BinnedDistDataset,
    BinnedDistStream,
    BinnedDistTokenizer,
    DistPriors,
    batch_from_items,
    make_eval_specs,
)
from model import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-value-bins", type=int, default=128)
    parser.add_argument("--num-param-bins", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-specs", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts/debug_run")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm-probs")
    return parser.parse_args()


def make_batch(iterator, batch_size, tokenizer):
    items = [next(iterator) for _ in range(batch_size)]
    inputs, labels, loss_mask = batch_from_items(items, tokenizer)
    return (
        jnp.asarray(inputs, dtype=jnp.int32),
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray(loss_mask, dtype=jnp.float32),
    )


def train_loss_fn(model, inputs, labels, loss_mask):
    logits = model(inputs, deterministic=False)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return (token_loss * loss_mask).sum() / loss_mask.sum()


def eval_loss_fn(model, inputs, labels, loss_mask):
    logits = model(inputs, deterministic=True)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return (token_loss * loss_mask).sum() / loss_mask.sum()


@nnx.jit
def train_step(model, optimizer, inputs, labels, loss_mask):
    loss, grads = nnx.value_and_grad(train_loss_fn)(model, inputs, labels, loss_mask)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, inputs, labels, loss_mask):
    return eval_loss_fn(model, inputs, labels, loss_mask)


def evaluate(model, dataset, tokenizer, batch_size):
    losses = []
    items = []
    for idx in range(len(dataset)):
        items.append(dataset[idx])
        if len(items) == batch_size or idx == len(dataset) - 1:
            inputs, labels, loss_mask = batch_from_items(items, tokenizer)
            loss = eval_step(
                model,
                jnp.asarray(inputs, dtype=jnp.int32),
                jnp.asarray(labels, dtype=jnp.int32),
                jnp.asarray(loss_mask, dtype=jnp.float32),
            )
            losses.append(float(loss))
            items = []
    return sum(losses) / len(losses)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    tokenizer = BinnedDistTokenizer(
        num_value_bins=args.num_value_bins,
        num_param_bins=args.num_param_bins,
    )
    priors = DistPriors()
    eval_specs = make_eval_specs(priors, args.eval_specs, args.seed + 1, tokenizer)
    eval_keys = {tokenizer.spec_key(spec) for spec in eval_specs}
    save_json(out_dir / "eval_specs.json", eval_specs)
    save_json(out_dir / "config.json", config | {"vocab_size": tokenizer.vocab_size})

    if args.wandb:
        wandb.init(project=args.wandb_project, config=config)

    print(f"JAX devices: {jax.devices()}")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rng=nnx.Rngs(args.seed),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(args.lr, weight_decay=args.weight_decay),
        wrt=nnx.Param,
    )

    train_stream = BinnedDistStream(
        priors,
        tokenizer,
        N=args.seq_len,
        seed=args.seed + 2,
        excluded_keys=eval_keys,
    )
    train_iter = iter(train_stream)
    eval_dataset = BinnedDistDataset(eval_specs, tokenizer, N=args.seq_len, seed=args.seed + 3)

    start = time.time()
    for step in range(1, args.steps + 1):
        inputs, labels, loss_mask = make_batch(train_iter, args.batch_size, tokenizer)
        loss = train_step(model, optimizer, inputs, labels, loss_mask)

        if step % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            tokens_per_sec = step * args.batch_size * args.seq_len / elapsed
            metrics = {
                "train/loss": float(loss),
                "train/tokens_per_sec": tokens_per_sec,
                "step": step,
            }
            print(metrics)
            if args.wandb:
                wandb.log(metrics, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            eval_loss = evaluate(model, eval_dataset, tokenizer, args.eval_batch_size)
            metrics = {"eval/loss": eval_loss, "step": step}
            print(metrics)
            if args.wandb:
                wandb.log(metrics, step=step)

    state = nnx.state(model, nnx.Param).to_pure_dict()
    (out_dir / "model.msgpack").write_bytes(serialization.to_bytes(state))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
