# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research question

When trained on samples from known probability distributions, do autoregressive vs. masked Transformers differ in (1) how accurately they reproduce the target distribution, (2) sample independence, and (3) how they internally represent mean / variance / modality / entropy? Motivation: *Large Language Models Are Bad Dice Players* — this project moves from behavioural prompting of frontier LLMs to controlled training + mechanistic analysis on small models.

Planned pipeline (per README):
1. Build distributions + evaluation metrics (`data.py`, `eval.py`).
2. Train a small causal transformer on samples from those distributions.
3. Vary masking / sampling strategies (consider diffusion-style).
4. Compare hidden representations against frontier LLMs on the same task.

## Current state

`src/llm_prob/` holds the small-model pipeline: char-level `tokenization.py`, distribution dataset builders in `data.py`, `TinyGPT` in `model.py`, `train.py`, and `eval.py` (`conditional_eval` reports parse rate, observed mu/sigma, KS vs. the target normal). LLM side: `llm_eval.py` mirrors `conditional_eval` for any HF causal LM, and `scripts/download_model.py` + `scripts/run_llm_eval.py` cover snapshot download and a first run on the cluster (Qwen2.5-3B as the starting model, with TransformerLens analysis planned next).

## Environment

- Python 3.12 (`.python-version`), managed with **uv**. `uv.lock` is checked in.
- Core deps: `torch`, `transformers`, `accelerate`, `huggingface_hub`, `transformer_lens`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `ipykernel`. No JAX.
- Run anything via `uv run …` so the right interpreter and lock are used.
- Add a dependency: `uv add <pkg>` (do not hand-edit `pyproject.toml` for deps).
- Pytest is configured (`pytest>=9.0.3` in the `dev` group); no linter/formatter yet.

## Framework choice

PyTorch throughout. No JAX/Flax. New model code should default to torch.

## Cluster / HF cache

Cluster home dirs are quota-limited — never let HF cache to `~/.cache`. Both `download_model.py` and `run_llm_eval.py` take `--cache-dir` and set `HF_HOME`/`HF_HUB_CACHE` so weights land on scratch.

## Vault counterpart

Per the user's global convention, each repo has a sibling `~/Documents/Vault/Notes/Lab/<name>/`. **There is no `Lab/llm_probs/` directory yet** — if the user launches with `--add-dir` pointing there, the session-start protocol (read `agenda.md` + latest `log/`) does not apply until that directory is created.
