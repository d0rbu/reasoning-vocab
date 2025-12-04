#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J rlvr-baseline
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --tasks-per-node 1
#SBATCH -t 48:00:00
#SBATCH -o baseline-%j
#SBATCH -e baseline-%j.err
#SBATCH -p h100

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

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

# Run baseline training (no reasoning vocabulary)
# Note: Hydra configs are in exp/conf/
# Override parameters with: key=value (e.g., training.learning_rate=1e-5)
srun uv run accelerate launch \
    --config_file accelerate_config/default.yaml \
    --multi_gpu \
    --num_processes $(($SLURM_JOB_NUM_NODES * 4)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    exp/grpo_train.py \
    model=baguettotron \
    training=grpo_baguettotron