# Configuration Examples

This file contains examples of common configuration overrides for different experimental scenarios.

## Basic Usage

### Test Run (Fast Iteration)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=quick_test \
    dataset.max_train_samples=50 \
    training.num_train_epochs=1 \
    training.per_device_train_batch_size=1 \
    training.logging_steps=5 \
    logging.mode=offline
```

### Baseline Training (No Reasoning Vocabulary)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=baseline_full \
    model.reasoning_vocab_size=0
```

### Reasoning Vocabulary Training

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning_full \
    model.reasoning_vocab_size=151646
```

## Hyperparameter Tuning

### Lower Learning Rate

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=lr_1e-6 \
    training.learning_rate=1e-6
```

### Different Batch Sizes

```bash
# Larger effective batch size (better for stability)
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=large_batch \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=8

# Smaller batch size (for limited memory)
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=small_batch \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=2
```

### Longer Training

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=long_training \
    training.num_train_epochs=10
```

## GRPO-Specific Settings

### More Generations (Better Exploration)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=more_generations \
    training.num_generations=8
```

### Longer Completions

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=longer_solutions \
    training.max_completion_length=1024
```

## Memory Optimization

### 8-bit Quantization

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=8bit \
    model.load_in_8bit=true
```

### 4-bit Quantization (QLoRA)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=4bit \
    model.load_in_4bit=true
```

### Gradient Checkpointing

```bash
# Already enabled by default, but can be toggled
uv run python rlvr_vocab/exp/grpo_train.py \
    training.gradient_checkpointing=false
```

## Dataset Variations

### Subsample Training Data

```bash
# Train on 10% of data
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=subsample_10pct \
    dataset.max_train_samples=4000  # ~10% of 40k total
```

### Custom Prompt Template

Edit `conf/dataset/deepscaler.yaml` or create a new config:

```yaml
# conf/dataset/deepscaler_instruct.yaml
name: "agentica-org/DeepScaleR-Preview-Dataset"
train_split: "train"
val_split: null
max_train_samples: null
max_val_samples: null

prompt_template: |
  [INST] You are a math expert. Solve the following problem step by step.
  
  Problem: {problem}
  
  Provide your final answer in the format: Answer: <your solution> [/INST]

answer_format: "latex"
```

Then use it:

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    dataset=deepscaler_instruct
```

## Logging Variations

### Offline WandB

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    logging.mode=offline
```

### Disable WandB

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    logging.enabled=false
```

### Custom WandB Project

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    logging.project=my-rlvr-experiments \
    logging.entity=my-wandb-team
```

### Add Tags

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    'logging.tags=[baseline,qwen,math]'
```

## Multi-Config Overrides

### Full Custom Experiment

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=custom_experiment_v1 \
    seed=1337 \
    model.reasoning_vocab_size=75000 \
    model.temperature=0.8 \
    training.learning_rate=3e-6 \
    training.num_train_epochs=5 \
    training.per_device_train_batch_size=4 \
    training.num_generations=6 \
    dataset.max_train_samples=20000 \
    'logging.tags=[custom,experiment,v1]' \
    logging.notes="Testing half-sized reasoning vocab"
```

## Reasoning Vocabulary Ablations

### Quarter-Size Reasoning Vocab

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning_quarter \
    model.reasoning_vocab_size=37911  # ~1/4 of 151646
```

### Double-Size Reasoning Vocab

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning_double \
    model.reasoning_vocab_size=303292  # 2x 151646
```

### Small Reasoning Vocab (Proof of Concept)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning_small \
    model.reasoning_vocab_size=1000
```

## Creating Config Groups

You can create new configuration files for common experimental setups:

### Example: Fast Iteration Config

Create `conf/training/fast_iter.yaml`:

```yaml
# Fast iteration config for quick testing
num_train_epochs: 1
learning_rate: 1e-5
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
num_generations: 2
max_prompt_length: 512
max_completion_length: 256
logging_steps: 5
save_steps: 100
eval_steps: 100
bf16: true
fp16: false
gradient_checkpointing: true
dataloader_num_workers: 2
remove_unused_columns: false
```

Then use it:

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    training=fast_iter \
    dataset.max_train_samples=100
```

### Example: Production Config

Create `conf/training/production.yaml`:

```yaml
# Production training config
num_train_epochs: 8
learning_rate: 3e-6
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_generations: 8
max_prompt_length: 1024
max_completion_length: 1024
logging_steps: 10
save_steps: 250
eval_steps: 250
save_total_limit: 5
bf16: true
fp16: false
gradient_checkpointing: true
dataloader_num_workers: 8
remove_unused_columns: false
```

Then use it:

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    training=production
```

