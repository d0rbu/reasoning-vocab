# GRPO Training Setup Complete ✅

This document summarizes the training infrastructure that has been set up for the RLVR vocabulary expansion project.

## What's Been Implemented

### 1. **Hydra Configuration Management** 
- ✅ Hierarchical configuration structure in `rlvr_vocab/exp/conf/`
- ✅ Separate config groups for: model, dataset, training, logging
- ✅ Command-line parameter overrides
- ✅ Organized model_kwargs and generation_kwargs

### 2. **Dataset Integration**
- ✅ Uses HuggingFace datasets directly (no bespoke implementation)
- ✅ Chat format with `tokenizer.apply_chat_template()`
- ✅ System prompt configuration
- ✅ Support for subsample training

### 3. **Reward Function**
- ✅ Uses TRL's built-in **`accuracy_reward`**
- ✅ Math verification for parseable answers
- ✅ Normalized text comparison for non-parseable answers
- ✅ Binary reward: 1.0 for correct, 0.0 for incorrect

### 4. **Training Script**
- ✅ Complete GRPO training script with Hydra integration
- ✅ Model and tokenizer loading with organized kwargs
- ✅ Chat template formatting
- ✅ WandB logging support
- ✅ Checkpoint management
- ✅ **Uses loguru for logging**
- ✅ **Imports torch as th**
- ✅ **Proper relative imports (no sys.path hacks)**

### 5. **SLURM Integration**
- ✅ Updated SLURM scripts with `module load WebProxy`
- ✅ Proper internet access for HuggingFace and WandB
- ✅ Baseline and reasoning vocabulary configurations
- ✅ Example Hydra overrides in comments

### 6. **Dependencies**
- ✅ `hydra-core>=1.3.0` - Configuration management
- ✅ `wandb>=0.16.0` - Experiment tracking
- ✅ `loguru>=0.7.0` - Structured logging

### 7. **Documentation**
- ✅ Comprehensive README in `rlvr_vocab/exp/README.md`
- ✅ Configuration examples in `conf/examples.md`
- ✅ Updated main README with quick start guide

## File Structure

```
rlvr_vocab/
├── exp/
│   ├── conf/               # Hydra configurations
│   │   ├── config.yaml     # Main config (seed=0)
│   │   ├── model/
│   │   │   └── qwen3_0.6b.yaml  # model_kwargs + generation_kwargs
│   │   ├── dataset/
│   │   │   └── deepscaler.yaml  # system_prompt for chat format
│   │   ├── training/
│   │   │   └── grpo_default.yaml
│   │   └── logging/
│   │       └── wandb.yaml
│   │
│   ├── grpo_train.py       # Main training script (uses TRL rewards)
│   └── README.md           # Training documentation
│
└── slurm/
    ├── train_baseline.sh   # SLURM script (no reasoning vocab)
    └── train_reasoning.sh  # SLURM script (with reasoning vocab)
```

## Key Design Decisions

### 1. No Bespoke Dataset Implementation
Uses HuggingFace datasets directly with chat template formatting:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": problem},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### 2. TRL's Built-in Rewards
Uses `accuracy_reward` from TRL instead of custom reward function:

```python
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    reward_funcs=accuracy_reward,  # TRL's built-in
)
```

The reward function automatically:
- Extracts answers (looks for `\\boxed{...}`)
- Compares with ground truth
- Returns 1.0 (correct) or 0.0 (incorrect)

### 3. Organized Configuration
Model config now has clear separation:

```yaml
# model_kwargs for loading
model_kwargs:
  torch_dtype: "bfloat16"  # fp32, fp16, bf16
  trust_remote_code: true
  load_in_8bit: false

# generation_kwargs for generation
generation_kwargs:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

### 4. Proper Imports
- Uses `import torch as th` (user's coding style)
- Uses `from loguru import logger` for structured logging
- No sys.path hacks - proper relative imports

### 5. Seed Set to 0
Default seed is now 0 (in `config.yaml`)

## Quick Start

### 1. Run Training

```bash
# Quick test (10 samples)
uv run python rlvr_vocab/exp/grpo_train.py \
    dataset.max_train_samples=10 \
    training.num_train_epochs=1
```

### 2. Submit to SLURM

```bash
# Baseline training
sbatch slurm/train_baseline.sh

# Reasoning vocabulary training
sbatch slurm/train_reasoning.sh
```

## Configuration Override Examples

```bash
# Override model dtype
python grpo_train.py model.model_kwargs.torch_dtype=fp16

# Override generation parameters
python grpo_train.py \
    model.generation_kwargs.temperature=0.8 \
    model.generation_kwargs.max_new_tokens=1024

# Custom system prompt
python grpo_train.py \
    dataset.system_prompt="You are an expert mathematician. Show your work."

# Multiple overrides
python grpo_train.py \
    exp_name=my_experiment \
    training.learning_rate=3e-6 \
    training.num_train_epochs=5 \
    model.reasoning_vocab_size=75000
```

## Dataset Format

DeepScaleR dataset fields:
- `problem`: Math problem text (LaTeX formatted)
- `answer`: Ground truth answer
- `solution`: Solution steps (may be empty)

These are automatically formatted into chat format:

```python
{
    "prompt": "<chat_template_formatted_prompt>",
    "answer": "4"  # Ground truth for accuracy_reward
}
```

## Reward Function Details

TRL's `accuracy_reward` function signature:

```python
def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],  # Ground truth answers
    **kwargs
) -> list[float]:
    # Returns list of rewards (1.0 or 0.0)
```

The function:
1. Extracts answer from completion (looks for `\\boxed{...}`)
2. If both are parseable → uses math verification
3. If not parseable → compares normalized text
4. Returns 1.0 if correct, 0.0 if incorrect

## Testing

All existing tests pass:
```bash
uv run pytest -v
# ============================== 16 passed in 0.04s ===============================
```

Code is formatted and linted:
```bash
uv run ruff check .
# All checks passed!
```

## Files Modified/Created

### New/Modified Files
- `rlvr_vocab/exp/grpo_train.py` - Refactored to use TRL rewards, loguru, torch as th
- `rlvr_vocab/exp/conf/config.yaml` - Changed seed to 0
- `rlvr_vocab/exp/conf/model/qwen3_0.6b.yaml` - Organized under model_kwargs/generation_kwargs
- `rlvr_vocab/exp/conf/dataset/deepscaler.yaml` - Simplified to system_prompt only
- `rlvr_vocab/exp/README.md` - Updated documentation
- `pyproject.toml` - Replaced sympy with loguru
- `TRAINING_SETUP.md` - This file

### Removed Files
- `rlvr_vocab/core/dataset.py` - No longer needed (use HF datasets directly)
- `rlvr_vocab/core/reward.py` - No longer needed (use TRL's accuracy_reward)
- `rlvr_vocab/exp/validate_setup.py` - Tests should go in test/ directory

## Summary

✅ Simplified architecture using TRL's built-in tools
✅ Chat format with tokenizer.apply_chat_template()
✅ Organized configuration (model_kwargs, generation_kwargs)
✅ Proper imports (torch as th, loguru logger)
✅ Seed set to 0
✅ No bespoke implementations - leverages TRL ecosystem
✅ All tests passing, code formatted and linted

The training infrastructure is ready for experiments! 🚀

