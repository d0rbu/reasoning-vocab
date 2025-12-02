#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rlvr-baseline
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=baseline-%j
#SBATCH --error=baseline-%j.err
#SBATCH --gres=gpu:pvc:8
#SBATCH --partition=pvc

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456
##SBATCH --mail-type=ALL
##SBATCH --mail-user=email_address

# Enable detailed logging
set -x

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NODE_RANK=$SLURM_NODEID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Load required modules (adjust for your cluster)
module load GCCcore/13.3.0 Python/3.12.3
module load WebProxy  # Required for internet access (HuggingFace, WandB)

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
export FI_PROVIDER=tcp

# Configure oneCCL worker threads
# Get the list of CPU cores assigned by SLURM to this job
# SLURM_JOB_CPUS_PER_NODE gives us the number of CPUs allocated
# We'll use the CPU list from taskset to get the actual core IDs
SLURM_CPUS=$(taskset -cp $$ | cut -d: -f2 | tr -d ' ')

# Set worker count to 1 to minimize resource usage
export CCL_WORKER_COUNT=1

# Use the first CPU from our SLURM allocation for the worker thread
# oneCCL expects a single CPU core, not a comma-separated list
FIRST_CPU=$(echo $SLURM_CPUS | cut -d',' -f1 | cut -d'-' -f1)
export CCL_WORKER_AFFINITY=$FIRST_CPU

echo "🔧 oneCCL Configuration:"
echo "   - SLURM assigned CPUs: $SLURM_CPUS"
echo "   - First CPU extracted: $FIRST_CPU"
echo "   - CCL_WORKER_COUNT: $CCL_WORKER_COUNT"
echo "   - CCL_WORKER_AFFINITY: $CCL_WORKER_AFFINITY"
echo ""

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
    --machine_rank $NODE_RANK \
    --rdzv_backend c10d \
    exp/grpo_train.py \
    model=baguettotron \
    training=grpo_baguettotron_grace