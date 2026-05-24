import argparse
import json

import numpy as np

from data import BinnedDistStream, BinnedDistTokenizer, DistPriors, batch_from_items


def token_name(token, tokenizer):
    special = {
        tokenizer.pad_token: "PAD",
        tokenizer.bos_token: "BOS",
        tokenizer.uniform_token: "TYPE_UNIFORM",
        tokenizer.gaussian_token: "TYPE_GAUSSIAN",
        tokenizer.bimodal_token: "TYPE_BIMODAL",
        tokenizer.param_null_token: "PARAM_NULL",
    }
    if token in special:
        return special[token]
    if tokenizer.sample_offset <= token < tokenizer.mu_offset:
        return f"X_BIN_{token - tokenizer.sample_offset}"
    if tokenizer.mu_offset <= token < tokenizer.sigma_offset:
        return f"MU_BIN_{token - tokenizer.mu_offset}"
    if tokenizer.sigma_offset <= token < tokenizer.w_offset:
        return f"SIGMA_BIN_{token - tokenizer.sigma_offset}"
    if tokenizer.w_offset <= token < tokenizer.vocab_size:
        return f"W_BIN_{token - tokenizer.w_offset}"
    return f"UNKNOWN_{token}"


def to_jsonable(item):
    return {
        "type": item["type"],
        "params": item["params"],
        "samples": np.round(item["samples"], 3).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-value-bins", type=int, default=16)
    parser.add_argument("--num-param-bins", type=int, default=8)
    parser.add_argument("--num-w-bins", type=int, default=8)
    args = parser.parse_args()

    tokenizer = BinnedDistTokenizer(
        num_value_bins=args.num_value_bins,
        num_param_bins=args.num_param_bins,
        num_w_bins=args.num_w_bins,
    )
    stream = BinnedDistStream(
        DistPriors(),
        tokenizer,
        N=args.seq_len,
        seed=args.seed,
    )
    iterator = iter(stream)
    items = [next(iterator) for _ in range(args.batch_size)]
    inputs, labels, loss_mask = batch_from_items(items, tokenizer)
    full_sequences = [tokenizer.encode_item(item) for item in items]

    print("tokenizer")
    print(
        json.dumps(
            {
                "vocab_size": tokenizer.vocab_size,
                "prefix_len": tokenizer.prefix_len,
                "sample_offset": tokenizer.sample_offset,
                "mu_offset": tokenizer.mu_offset,
                "sigma_offset": tokenizer.sigma_offset,
                "w_offset": tokenizer.w_offset,
            },
            indent=2,
        )
    )
    print()

    for idx, item in enumerate(items):
        print(f"item[{idx}] raw")
        print(json.dumps(to_jsonable(item), indent=2))
        print(f"item[{idx}] full_sequence_ids")
        print(full_sequences[idx].tolist())
        print(f"item[{idx}] full_sequence_names")
        print([token_name(int(t), tokenizer) for t in full_sequences[idx]])
        print()

    print("batch inputs")
    print(inputs.tolist())
    print("batch labels")
    print(labels.tolist())
    print("batch loss_mask")
    print(loss_mask.astype(int).tolist())


if __name__ == "__main__":
    main()
