"""llm_prob: probing how small Transformers represent simple probability distributions."""

from .data import (
    collate,
    make_continuous_mu_dataset,
    make_fixed_dataset,
    make_mean_sweep_dataset,
)
from .eval import conditional_eval, generate, parse_samples, run_eval
from .model import TinyGPT
from .plots import (
    plot_conditional_eval,
    plot_entropy_by_position,
    plot_eval_distribution,
)
from .tokenization import (
    BOS,
    EOS,
    ITOS,
    P,
    PAD,
    SEP,
    STOI,
    VOCAB,
    build_sequence,
    encode_number,
    fmt,
)
from .train import train, train_step

__all__ = [
    'BOS', 'EOS', 'ITOS', 'P', 'PAD', 'SEP', 'STOI', 'VOCAB',
    'TinyGPT',
    'build_sequence', 'encode_number', 'fmt',
    'collate', 'make_fixed_dataset', 'make_mean_sweep_dataset', 'make_continuous_mu_dataset',
    'train', 'train_step',
    'generate', 'parse_samples', 'run_eval', 'conditional_eval',
    'plot_eval_distribution', 'plot_entropy_by_position', 'plot_conditional_eval',
]
