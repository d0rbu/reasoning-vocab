#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rlvr-reasoning
#SBATCH --time=48:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=reasoning-%j
#SBATCH --error=reasoning-%j.err
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456
##SBATCH --mail-type=ALL
##SBATCH --mail-user=email_address

# Enable detailed logging
set -x

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Load required modules (adjust for your cluster)
module load GCCcore/13.3.0 Python/3.12.3
module load WebProxy  # Required for internet access (HuggingFace, WandB)

# Change to the project directory
cd $SCRATCH/reasoning-vocab

# Activate environment
source .venv/bin/activate

# Run training with reasoning vocabulary (reasoning_vocab_size = vocab_size)
# For Qwen3-0.6B, vocab_size is approximately 151646
# Note: Hydra configs are in exp/conf/
# Override parameters with: key=value (e.g., training.learning_rate=1e-5)
uv run accelerate launch --num_processes 4 \
    --num_machines 2 \
    exp/grpo_train.py \
    exp_name=reasoning_vocab_run \
    model=baguettotron \
    model.reasoning_vocab_size=65536
