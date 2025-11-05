# 🧩 RLVR + Vocabulary Expansion for LLM Post-Training

This project explores **Reinforcement Learning with Verified Rewards (RLVR)** for **LLM post-training**, extending a pretrained causal language model (such as `Qwen3ForCausalLM`) with an expanded token vocabulary for internal reasoning.

## 🧠 Project Overview

The core innovation is introducing a **"reasoning vocabulary"** — additional embeddings and unembeddings used **only within `<reasoning>` … `</reasoning>` blocks**.
When reasoning tokens are active, the model uses a **2× vocabulary** where the size of the reasoning vocabulary `n_r` is equal to the standard vocabulary `n_s`:

* Standard vocabulary (`n_s` tokens) for normal text
* Reasoning vocabulary (`n_r` new tokens) for internal thought

## 🏗️ Directory Structure

```
rlvr_vocab/
├── core/                  # Core model, token, and training logic
├── exp/                   # Experiment scripts and configs
├── viz/                   # Visualization scripts
├── model/                 # Saved model checkpoints
├── data/                  # Datasets (GSM8K, MATH, etc.)
├── out/                   # Training outputs (logs, metrics)
├── fig/                   # Saved figures
├── slurm/                 # Scripts to run experiments on SLURM
└── test/                  # Pytest unit tests
```

## ⚙️ Setup

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create environment and install dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 3. Set up pre-commit hooks

```bash
pre-commit install
```

## 🚀 Quick Start

### Validate Setup

```bash
# Check that everything is configured correctly
uv run python rlvr_vocab/exp/validate_setup.py
```

### Run Training

```bash
# Quick test run (10 samples, 1 epoch)
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=test_run \
    dataset.max_train_samples=10 \
    training.num_train_epochs=1

# Baseline training (no reasoning vocabulary)
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=baseline

# With reasoning vocabulary
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning \
    model.reasoning_vocab_size=151646
```

### SLURM Training

```bash
# Baseline training
sbatch slurm/train_baseline.sh

# Reasoning vocabulary training  
sbatch slurm/train_reasoning.sh
```

See [rlvr_vocab/exp/README.md](rlvr_vocab/exp/README.md) for detailed training documentation.

## 🧪 Development

### Run tests

```bash
uv run pytest
```

### Linting and formatting

```bash
uv run ruff check --fix .
uv run ruff format .
```

### Type checking

```bash
uvx ty check .
```

## 📘 Documentation and References

* **Hugging Face TRL:** https://github.com/huggingface/trl
* **Qwen3 Model Reference:** https://github.com/huggingface/transformers
* **uv Package Manager:** https://github.com/astral-sh/uv
* **Ruff Linter Docs:** https://docs.astral.sh/ruff
* **ty:** https://docs.astral.sh/ty

## 📄 License

MIT License
