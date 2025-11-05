# GRPO Training Setup Complete ✅

This document summarizes the training infrastructure that has been set up for the RLVR vocabulary expansion project.

## What's Been Implemented

### 1. **Hydra Configuration Management** 
- ✅ Hierarchical configuration structure in `rlvr_vocab/exp/conf/`
- ✅ Separate config groups for: model, dataset, training, logging
- ✅ Command-line parameter overrides
- ✅ Example configurations in `conf/examples.md`

### 2. **Dataset Integration**
- ✅ Dataset loader for `agentica-org/DeepScaleR-Preview-Dataset`
- ✅ Preprocessing pipeline for math problems
- ✅ Flexible prompt templating
- ✅ Answer normalization and extraction

### 3. **Reward Function**
- ✅ Multi-method answer verification:
  - Exact string matching (normalized)
  - Numeric comparison (with tolerance)
  - Symbolic comparison (SymPy)
  - LaTeX parsing
- ✅ Binary reward: 1.0 for correct, 0.0 for incorrect

### 4. **Training Script**
- ✅ Complete GRPO training script with Hydra integration
- ✅ Model and tokenizer loading
- ✅ Dataset preprocessing
- ✅ Reward function integration
- ✅ WandB logging support
- ✅ Checkpoint management

### 5. **SLURM Integration**
- ✅ Updated SLURM scripts with `module load WebProxy`
- ✅ Proper internet access for HuggingFace and WandB
- ✅ Baseline and reasoning vocabulary configurations
- ✅ Example Hydra overrides in comments

### 6. **Dependencies**
- ✅ Added `hydra-core>=1.3.0`
- ✅ Added `wandb>=0.16.0`
- ✅ Added `sympy>=1.12` for symbolic math

### 7. **Documentation**
- ✅ Comprehensive README in `rlvr_vocab/exp/README.md`
- ✅ Configuration examples in `conf/examples.md`
- ✅ Updated main README with quick start guide
- ✅ Validation script for setup checking

## File Structure

```
rlvr_vocab/
├── core/
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── reward.py           # Math correctness reward function
│
├── exp/
│   ├── conf/               # Hydra configurations
│   │   ├── config.yaml     # Main config
│   │   ├── model/
│   │   │   └── qwen3_0.6b.yaml
│   │   ├── dataset/
│   │   │   └── deepscaler.yaml
│   │   ├── training/
│   │   │   └── grpo_default.yaml
│   │   └── logging/
│   │       └── wandb.yaml
│   │
│   ├── grpo_train.py       # Main training script
│   ├── validate_setup.py   # Setup validation script
│   └── README.md           # Training documentation
│
└── slurm/
    ├── train_baseline.sh   # SLURM script (no reasoning vocab)
    └── train_reasoning.sh  # SLURM script (with reasoning vocab)
```

## Quick Start

### 1. Validate Setup

```bash
uv run python rlvr_vocab/exp/validate_setup.py
```

This checks:
- Dependencies are installed
- Dataset can be loaded
- Reward function works
- Hydra configuration is valid

### 2. Test Run (10 samples, 1 epoch)

```bash
uv run python rlvr_vocab/exp/grpo_train.py \
    exp_name=test_run \
    dataset.max_train_samples=10 \
    training.num_train_epochs=1 \
    logging.mode=offline
```

### 3. Submit to SLURM

```bash
# Baseline training
sbatch slurm/train_baseline.sh

# Reasoning vocabulary training
sbatch slurm/train_reasoning.sh
```

## Configuration Override Examples

The Hydra configuration system allows flexible command-line overrides:

```bash
# Override learning rate
python grpo_train.py training.learning_rate=1e-5

# Change model
python grpo_train.py model.name=Qwen/Qwen2.5-1.5B

# Subsample dataset
python grpo_train.py dataset.max_train_samples=5000

# Multiple overrides
python grpo_train.py \
    exp_name=my_experiment \
    training.learning_rate=3e-6 \
    training.num_train_epochs=5 \
    model.reasoning_vocab_size=75000 \
    'logging.tags=[custom,ablation]'
```

See `rlvr_vocab/exp/conf/examples.md` for more examples.

## Dataset Format

The DeepScaleR dataset contains:
- `problem`: Math problem text (LaTeX formatted)
- `answer`: Ground truth answer
- `solution`: Solution steps (may be empty)

Example:
```json
{
  "problem": "Solve for x: 2x + 5 = 13",
  "answer": "4",
  "solution": "2x + 5 = 13\n2x = 8\nx = 4"
}
```

## Reward Function

The reward function extracts and verifies answers:

1. **Extract answer from completion**:
   - Look for `\\boxed{...}` pattern
   - Look for "Answer:" pattern
   - Look for "The answer is" pattern
   - Find last equation with `=`
   - Use last non-empty line

2. **Compare with ground truth**:
   - Exact match (normalized)
   - Numeric comparison (tolerance: 1e-6)
   - Symbolic comparison (SymPy)
   - LaTeX parsing

3. **Return binary reward**:
   - 1.0 if correct
   - 0.0 if incorrect

## WandB Integration

The training script automatically logs to WandB:

```bash
# Login to WandB
wandb login

# Run with WandB (online mode)
python grpo_train.py

# Offline mode (sync later)
python grpo_train.py logging.mode=offline

# Disable WandB
python grpo_train.py logging.enabled=false
```

Logged metrics include:
- Loss and learning rate
- Reward statistics
- Generation samples
- Model checkpoints (optional)

## SLURM Configuration

Both SLURM scripts now include:
- `module load WebProxy` for internet access
- Proper environment variables for distributed training
- Example Hydra overrides in comments

Key SLURM settings:
- 1 H100 GPU
- 128GB RAM
- 16 CPU cores
- 48-hour time limit

## Next Steps

1. **Validate the setup**:
   ```bash
   uv run python rlvr_vocab/exp/validate_setup.py
   ```

2. **Run a quick test**:
   ```bash
   uv run python rlvr_vocab/exp/grpo_train.py \
       dataset.max_train_samples=10 \
       training.num_train_epochs=1
   ```

3. **Submit baseline job**:
   ```bash
   sbatch slurm/train_baseline.sh
   ```

4. **Monitor in WandB**:
   - Check training progress
   - View sample generations
   - Compare experiments

## Troubleshooting

### Out of Memory
- Reduce batch size: `training.per_device_train_batch_size=1`
- Increase gradient accumulation: `training.gradient_accumulation_steps=8`
- Enable 8-bit loading: `model.load_in_8bit=true`

### Dataset Download Issues
- Check internet access (SLURM: verify WebProxy module loaded)
- Login to HuggingFace: `huggingface-cli login`

### WandB Connection Issues
- Use offline mode: `logging.mode=offline`
- Disable WandB: `logging.enabled=false`

## Files Modified/Created

### New Files
- `rlvr_vocab/core/dataset.py` - Dataset loading
- `rlvr_vocab/core/reward.py` - Reward function
- `rlvr_vocab/exp/grpo_train.py` - Main training script
- `rlvr_vocab/exp/validate_setup.py` - Validation script
- `rlvr_vocab/exp/conf/config.yaml` - Main Hydra config
- `rlvr_vocab/exp/conf/model/qwen3_0.6b.yaml` - Model config
- `rlvr_vocab/exp/conf/dataset/deepscaler.yaml` - Dataset config
- `rlvr_vocab/exp/conf/training/grpo_default.yaml` - Training config
- `rlvr_vocab/exp/conf/logging/wandb.yaml` - Logging config
- `rlvr_vocab/exp/conf/examples.md` - Configuration examples
- `rlvr_vocab/exp/README.md` - Training documentation
- `TRAINING_SETUP.md` - This file

### Modified Files
- `pyproject.toml` - Added hydra-core, wandb, sympy
- `slurm/train_baseline.sh` - Added WebProxy, updated for Hydra
- `slurm/train_reasoning.sh` - Added WebProxy, updated for Hydra
- `README.md` - Added quick start section

### Removed Files
- `rlvr_vocab/exp/configs/` - Replaced with `conf/` for Hydra

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

uv run ruff format .
# 4 files reformatted, 15 files left unchanged
```

## Summary

✅ Complete Hydra-based configuration system
✅ Dataset loading and preprocessing
✅ Multi-method reward function for math verification
✅ Full GRPO training script with WandB integration
✅ SLURM scripts with internet access (WebProxy)
✅ Comprehensive documentation and examples
✅ Validation script for setup checking
✅ All tests passing, code formatted and linted

The training infrastructure is ready for experiments! 🚀

