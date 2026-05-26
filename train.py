import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import yaml
from flax import nnx, serialization

from data import (
    BinnedDistDataset,
    BinnedDistStream,
    BinnedDistTokenizer,
    DigitDistTokenizer,
    DistPriors,
    batch_from_items,
    empirical_binned_probs,
    make_eval_specs,
    true_binned_probs,
)
from model import Transformer


DEFAULT_CONFIG = {
    "tokenizer": "digit",
    "steps": 20000,
    "batch_size": 1,
    "seq_len": 512,
    "num_value_bins": 512,
    "num_param_bins": 128,
    "num_w_bins": 128,
    "model_dim": 128,
    "hidden_dim": 512,
    "num_heads": 4,
    "num_layers": 4,
    "dropout": 0.1,
    "lr": 6e-4,
    "warmup_steps": 1000,
    "min_lr": 6e-5,
    "weight_decay": 1e-2,
    "eval_specs": 128,
    "eval_batch_size": 1,
    "eval_every": 500,
    "generate_every": 1000,
    "generate_specs": 4,
    "generate_samples": 64,
    "temperature": 1.0,
    "top_k": 0,
    "checkpoint_every": 5000,
    "log_every": 50,
    "seed": 0,
    "out_dir": "artifacts/digit_decoder_512",
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


def make_tokenizer(args):
    kwargs = {
        "num_value_bins": args.num_value_bins,
        "num_param_bins": args.num_param_bins,
        "num_w_bins": args.num_w_bins,
    }
    if args.tokenizer == "binned":
        return BinnedDistTokenizer(**kwargs)
    if args.tokenizer == "digit":
        return DigitDistTokenizer(**kwargs)
    raise ValueError(f"unknown tokenizer: {args.tokenizer!r}")


def save_checkpoint(model, out_dir, name):
    state = nnx.state(model, nnx.Param).to_pure_dict()
    (out_dir / name).write_bytes(serialization.to_bytes(state))


def evaluate(model, dataset, tokenizer, batch_size):
    losses = []
    items = []
    oracle_losses = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        items.append(item)
        if isinstance(tokenizer, BinnedDistTokenizer):
            probs = true_binned_probs(item, tokenizer)
            sample_bins = tokenizer.encode_samples(item["samples"]) - tokenizer.sample_offset
            oracle_losses.extend((-jnp.log(jnp.asarray(probs)[sample_bins])).tolist())
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
    metrics = {"eval/loss": sum(losses) / len(losses)}
    if oracle_losses:
        oracle_ce = float(sum(oracle_losses) / len(oracle_losses))
        metrics.update(
            {
                "eval/oracle_ce": oracle_ce,
                "eval/uniform_ce": float(jnp.log(tokenizer.num_value_bins)),
                "eval/excess_ce": metrics["eval/loss"] - oracle_ce,
            }
        )
    return metrics


def _sample_token(logits, key, temperature=1.0, top_k=0, allowed_tokens=None):
    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature
    if allowed_tokens is not None:
        mask = jnp.full(logits.shape, -jnp.inf)
        mask = mask.at[jnp.asarray(allowed_tokens, dtype=jnp.int32)].set(0.0)
        logits = logits + mask
    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        values, _ = jax.lax.top_k(logits, k)
        logits = jnp.where(logits < values[-1], -jnp.inf, logits)
    return int(jax.random.categorical(key, logits))


def generate_tokens(model, tokenizer, spec, n_samples, rng_key, temperature=1.0, top_k=0):
    prefix = tokenizer.encode_spec(spec)
    tokens = list(prefix)
    if isinstance(tokenizer, DigitDistTokenizer):
        new_tokens = n_samples * tokenizer.sample_token_width
        allowed_tokens = None
    else:
        new_tokens = n_samples
        allowed_tokens = jnp.arange(tokenizer.sample_offset, tokenizer.mu_offset)

    for _ in range(new_tokens):
        rng_key, sample_key = jax.random.split(rng_key)
        inputs = jnp.asarray([tokens], dtype=jnp.int32)
        logits = model(inputs, deterministic=True)[0, -1]
        tokens.append(_sample_token(logits, sample_key, temperature, top_k, allowed_tokens))
    return jnp.asarray(tokens[len(prefix):], dtype=jnp.int32), rng_key


def distribution_metrics(spec, generated_values, valid, tokenizer):
    eps = 1e-12
    target = true_binned_probs(spec, tokenizer)
    generated = empirical_binned_probs(generated_values, tokenizer, eps=eps)
    target = jnp.asarray(target + eps)
    generated = jnp.asarray(generated + eps)
    target = target / target.sum()
    generated = generated / generated.sum()
    mixture = 0.5 * (target + generated)

    centers = jnp.asarray(tokenizer.bin_centers())
    target_mean = float((target * centers).sum())
    gen_mean = float((generated * centers).sum())
    target_var = float((target * (centers - target_mean) ** 2).sum())
    gen_var = float((generated * (centers - gen_mean) ** 2).sum())
    bin_width = (tokenizer.value_range[1] - tokenizer.value_range[0]) / tokenizer.num_value_bins

    finite_values = np.asarray(generated_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) > 1:
        centered = finite_values - finite_values.mean()
        denom = np.sum(centered ** 2)
        autocorr = np.sum(centered[:-1] * centered[1:]) / denom if denom > 0 else 0.0
        duplicate_rate = np.mean(finite_values[1:] == finite_values[:-1])
    else:
        autocorr = 0.0
        duplicate_rate = 0.0

    return {
        "kl_target_generated": float(jnp.sum(target * jnp.log(target / generated))),
        "js_divergence": float(0.5 * jnp.sum(target * jnp.log(target / mixture)) + 0.5 * jnp.sum(generated * jnp.log(generated / mixture))),
        "total_variation": float(0.5 * jnp.sum(jnp.abs(target - generated))),
        "wasserstein": float(jnp.sum(jnp.abs(jnp.cumsum(target) - jnp.cumsum(generated))) * bin_width),
        "mean_error": gen_mean - target_mean,
        "variance_error": gen_var - target_var,
        "invalid_rate": float(1.0 - np.asarray(valid, dtype=float).mean()) if len(valid) else 1.0,
        "lag1_autocorr": float(autocorr),
        "duplicate_rate": float(duplicate_rate),
    }


def evaluate_generation(model, specs, tokenizer, args, step):
    key = jax.random.PRNGKey(args.seed + 10_000 + step)
    metric_sums = {}
    n_specs = min(args.generate_specs, len(specs))
    for spec in specs[:n_specs]:
        generated_tokens, key = generate_tokens(
            model,
            tokenizer,
            spec,
            args.generate_samples,
            key,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        values, valid = tokenizer.decode_samples(generated_tokens)
        spec_metrics = distribution_metrics(spec, values, valid, tokenizer)
        for name, value in spec_metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + value
    return {f"gen/{name}": value / n_specs for name, value in metric_sums.items()}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    tokenizer = make_tokenizer(args)
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
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=max(args.steps, args.warmup_steps + 1),
        end_value=args.min_lr,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(schedule, weight_decay=args.weight_decay),
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
    total_tokens = 0
    for step in range(1, args.steps + 1):
        inputs, labels, loss_mask = make_batch(train_iter, args.batch_size, tokenizer)
        total_tokens += inputs.size
        loss = train_step(model, optimizer, inputs, labels, loss_mask)

        if step % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            tokens_per_sec = total_tokens / elapsed
            metrics = {
                "train/loss": float(loss),
                "train/lr": float(schedule(step)),
                "train/tokens_per_sec": tokens_per_sec,
                "step": step,
            }
            print(metrics)
            if args.wandb:
                wandb.log(metrics, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, eval_dataset, tokenizer, args.eval_batch_size)
            metrics["step"] = step
            print(metrics)
            if args.wandb:
                wandb.log(metrics, step=step)

        if args.generate_every > 0 and (step % args.generate_every == 0 or step == args.steps):
            metrics = evaluate_generation(model, eval_specs, tokenizer, args, step)
            metrics["step"] = step
            print(metrics)
            save_json(out_dir / f"generation_metrics_step_{step}.json", metrics)
            if args.wandb:
                wandb.log(metrics, step=step)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, out_dir, f"model_step_{step}.msgpack")

    save_checkpoint(model, out_dir, "model.msgpack")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
