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

Scaffolding only. `main.py` is a hello-world; `data.py` and `eval.py` are empty; `utils.py` contains a stub `generate_discrete_guassian(mu, sigma, rng=(0,100), N=512)` (note: typo in name, no body). When extending, expect to be defining the module conventions for the first time — no prior structure to match.

## Environment

- Python 3.12 (`.python-version`), managed with **uv**. `uv.lock` is checked in.
- Core deps: `jax`, `numpy`, `matplotlib`, `tqdm`, `ipykernel`.
- Run anything via `uv run …` so the right interpreter and lock are used. Examples:
  - `uv run python main.py`
  - `uv run python -c "from utils import generate_discrete_guassian; ..."`
- Add a dependency: `uv add <pkg>` (do not hand-edit `pyproject.toml` for deps).
- No tests, linter, or formatter configured yet. Don't invent a `pytest`/`ruff` command pretending it exists — set one up explicitly if needed.

## Framework choice

`jax` is the listed numerics dep, not torch. Default new model / training code to JAX (+ likely `flax`/`optax` when added) unless the user says otherwise.

## Vault counterpart

Per the user's global convention, each repo has a sibling `~/Documents/Vault/Notes/Lab/<name>/`. **There is no `Lab/llm_probs/` directory yet** — if the user launches with `--add-dir` pointing there, the session-start protocol (read `agenda.md` + latest `log/`) does not apply until that directory is created.
