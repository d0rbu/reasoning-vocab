# GRPO Training for RLVR Vocabulary Expansion

This directory contains the training scripts and configurations for GRPO (Group Relative Policy Optimization) training with optional reasoning vocabulary expansion.

## Quick Start

### Local Training (CPU/Single GPU)

```bash
# Baseline training (no reasoning vocabulary)
uv run python rlvr_vocab/exp/grpo_train.py

# With reasoning vocabulary
uv run python rlvr_vocab/exp/grpo_train.py model.reasoning_vocab_size=151646

# Override any parameter
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=my_experiment \
    training.learning_rate=1e-5 \
    training.num_train_epochs=2 \
    dataset.max_train_samples=1000
```

### SLURM Training

```bash
# Baseline training
sbatch slurm/train_baseline.sh

# Reasoning vocabulary training
sbatch slurm/train_reasoning.sh
```

## Configuration Structure

Configurations use [Hydra](https://hydra.cc/) for hierarchical composition:

```
conf/
├── config.yaml              # Main config (composes others)
├── model/
│   └── qwen3_0.6b.yaml     # Model settings
├── dataset/
│   └── deepscaler.yaml     # Dataset settings
├── training/
│   └── grpo_default.yaml   # Training hyperparameters
└── logging/
    └── wandb.yaml          # Logging configuration
```

## Configuration Override Examples

### Override via Command Line

```bash
# Change learning rate
python grpo_train.py training.learning_rate=5e-6

# Change model
python grpo_train.py model.name=Qwen/Qwen2.5-1.5B

# Change dataset size
python grpo_train.py dataset.max_train_samples=5000

# Multiple overrides
python grpo_train.py \
    exp_name=fast_test \
    training.num_train_epochs=1 \
    training.per_device_train_batch_size=1 \
    dataset.max_train_samples=100
```

### Use Different Config Groups

```bash
# Use a different model config (create new yaml in conf/model/)
python grpo_train.py model=qwen3_1.5b

# Use a different dataset (create new yaml in conf/dataset/)
python grpo_train.py dataset=gsm8k
```

## Key Parameters

### Model Configuration
- `model.name`: HuggingFace model identifier
- `model.reasoning_vocab_size`: Size of reasoning vocabulary (0 = baseline)
- `model.model_kwargs.torch_dtype`: Precision (fp32, fp16, bf16)
- `model.model_kwargs.load_in_8bit`: Enable 8-bit quantization
- `model.generation_kwargs.max_new_tokens`: Max tokens to generate
- `model.generation_kwargs.temperature`: Sampling temperature

### Training Configuration
- `training.learning_rate`: Learning rate (default: 5e-6)
- `training.num_train_epochs`: Number of epochs (default: 4)
- `training.per_device_train_batch_size`: Batch size per device (default: 2)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `training.num_generations`: GRPO generations per prompt (default: 4)

### Dataset Configuration
- `dataset.name`: HuggingFace dataset identifier
- `dataset.max_train_samples`: Limit training samples (null = all)
- `dataset.system_prompt`: System prompt for chat format

### Logging Configuration
- `logging.enabled`: Enable/disable WandB (default: true)
- `logging.project`: WandB project name (default: "rlvr-vocab")
- `logging.mode`: online, offline, or disabled

## Weights & Biases Integration

The training script automatically logs to WandB. Make sure you're logged in:

```bash
wandb login
```

To disable WandB:

```bash
python grpo_train.py logging.enabled=false
```

Or use offline mode:

```bash
python grpo_train.py logging.mode=offline
```

## Dataset Format

The training script uses **conversational format** with the tokenizer's chat template:

```python
messages = [
    {"role": "system", "content": "You are a helpful math assistant..."},
    {"role": "user", "content": "Solve: 2x + 5 = 13"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Supported Datasets
- **agentica-org/DeepScaleR-Preview-Dataset**: Math problems (default)
  - Fields: `problem`, `answer`, `solution`
- Custom datasets can be added by creating new config files

## Reward Function

The training uses TRL's built-in **`accuracy_reward`** function which:

1. Extracts the answer from the completion (looks for `\\boxed{...}` pattern)
2. Compares with ground truth using:
   - Math verification (if both are parseable)
   - Normalized text comparison (if not parseable)
3. Returns 1.0 for correct, 0.0 for incorrect

Reference: https://huggingface.co/docs/trl/v0.24.0/rewards

## Output Structure

Training outputs are saved to `./out/{exp_name}/`:
- `checkpoint-{step}/`: Model checkpoints
- `final_model/`: Final trained model
- Logs and metrics (if WandB is enabled)

## Example: Full Training Pipeline

```bash
# 1. Test configuration (fast iteration)
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=test_run \
    dataset.max_train_samples=100 \
    training.num_train_epochs=1 \
    logging.mode=offline

# 2. Baseline training
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=baseline_full \
    model.reasoning_vocab_size=0

# 3. Reasoning vocabulary training
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=reasoning_full \
    model.reasoning_vocab_size=151646

# 4. Compare results in WandB
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `training.per_device_train_batch_size`
- Increase `training.gradient_accumulation_steps`
- Enable `model.model_kwargs.load_in_8bit=true`
- Ensure `training.gradient_checkpointing=true`

### Slow Training
- Increase `training.per_device_train_batch_size`
- Decrease `training.num_generations`
- Use mixed precision: `training.bf16=true`

### Dataset Download Issues
- Ensure internet access (SLURM: check `module load WebProxy`)
- Check HuggingFace credentials: `huggingface-cli login`

### WandB Connection Issues
- Check internet access
- Use offline mode: `logging.mode=offline`
- Disable WandB: `logging.enabled=false`

## Advanced Usage

### Custom System Prompts

Edit `conf/dataset/deepscaler.yaml`:

```yaml
system_prompt: "You are an expert mathematician. Show all your work and explain each step clearly."
```

### Multiple Reward Functions

You can combine multiple reward functions (see TRL docs):

```python
from trl.rewards import accuracy_reward, think_format_reward

# In grpo_train.py, modify:
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    reward_funcs=[accuracy_reward, think_format_reward],  # Multiple rewards
)
```

### Multi-GPU Training

The SLURM scripts are configured for single GPU. For multi-GPU:

```bash
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
```

Then update the srun command to use torchrun or accelerate launch.

