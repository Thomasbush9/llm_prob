import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
import yaml
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


DEFAULT_CONFIG = {
    "steps": 1000,
    "batch_size": 32,
    "seq_len": 128,
    "num_value_bins": 128,
    "num_param_bins": 64,
    "model_dim": 64,
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.0,
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "eval_specs": 64,
    "eval_batch_size": 32,
    "eval_every": 100,
    "log_every": 10,
    "seed": 0,
    "out_dir": "artifacts/debug_run",
    "wandb": False,
    "wandb_project": "llm-probs",
}


def add_config_arg(parser, name, value):
    arg_name = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        parser.add_argument(arg_name, action="store_true", default=None)
    else:
        parser.add_argument(arg_name, type=type(value), default=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    for name, value in DEFAULT_CONFIG.items():
        add_config_arg(parser, name, value)

    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            yaml_config = yaml.safe_load(f) or {}
        unknown_keys = set(yaml_config) - set(DEFAULT_CONFIG)
        if unknown_keys:
            raise ValueError(f"unknown config keys in {config_path}: {sorted(unknown_keys)}")
        config.update(yaml_config)

    cli_config = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    config.update(cli_config)
    config["config"] = str(config_path)
    return argparse.Namespace(**config)


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
