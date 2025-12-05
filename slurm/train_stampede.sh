#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J rlvr-grpo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o grpo-%j
#SBATCH -e grpo-%j.err
#SBATCH -p h100

# Enable detailed logging
set -x

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

export CUDA_LAUNCH_BLOCKING=1

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
    --num_processes $(($SLURM_JOB_NUM_NODES * 2)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --rdzv_backend static \
    --gpu_ids 0,1 \
    exp/grpo_train.py \
    model=baguettotron \
    training=grpo_baguettotron

srun uv run accelerate launch \
    --config_file accelerate_config/default.yaml \
    --num_processes $(($SLURM_JOB_NUM_NODES * 2)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --rdzv_backend static \
    --gpu_ids 2,3 \
    exp/grpo_train.py \
    exp_name=reasoning_vocab_run \
    model=baguettotron \
    model.reasoning_vocab_size=65536 \
    training=grpo_baguettotron