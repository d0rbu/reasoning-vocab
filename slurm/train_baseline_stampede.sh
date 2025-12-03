#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rlvr-baseline
#SBATCH --time=48:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=baseline-%j
#SBATCH --error=baseline-%j.err
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=h100

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

# Change to the project directory
cd $SCRATCH/reasoning-vocab

# Activate environment
source .venv/bin/activate

# Run baseline training (no reasoning vocabulary)
# Note: Hydra configs are in exp/conf/
# Override parameters with: key=value (e.g., training.learning_rate=1e-5)
srun uv run accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --rdzv_backend c10d \
    exp/grpo_train.py \
    model=baguettotron \
    training=grpo_baguettotron