#!/bin/bash
#SBATCH --job-name=rlvr-baseline
#SBATCH --output=out/slurm-%j.out
#SBATCH --error=out/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# TODO: Adjust SLURM parameters for your cluster

# Load modules (if needed)
# module load python/3.11
# module load cuda/12.1

# Activate environment
source .venv/bin/activate

# Run training
python rlvr_vocab/exp/grpo_train.py \
    --config rlvr_vocab/exp/configs/grpo_config.yaml \
    --model-config rlvr_vocab/exp/configs/qwen3_reasoning.yaml \
    --dataset-config rlvr_vocab/exp/configs/dataset_config.yaml

