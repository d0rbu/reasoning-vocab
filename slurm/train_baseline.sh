#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rlvr-baseline
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=out/baseline-%j.out
#SBATCH --error=out/baseline-%j.err
#SBATCH --gres=gpu:h100:1
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

# Change to the project directory
cd $SCRATCH/reasoning-vocab

# Activate environment
source .venv/bin/activate

# Run baseline training (no reasoning vocabulary)
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    uv run rlvr_vocab/exp/grpo_train.py \
    --config rlvr_vocab/exp/configs/grpo_config.yaml \
    --model-config rlvr_vocab/exp/configs/qwen3_reasoning.yaml \
    --dataset-config rlvr_vocab/exp/configs/dataset_config.yaml

