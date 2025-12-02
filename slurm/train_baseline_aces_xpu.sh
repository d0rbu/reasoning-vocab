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

# Load Intel MPI for oneCCL (required even for single-node)
# Check what Intel MPI modules are available: module avail intel-mpi or impi
module load intel-mpi/2021.14 || module load impi/2021.14 || echo "Warning: Intel MPI module not found"

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
export USE_XETLA=OFF
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

# For single-node multi-GPU: configure oneCCL for local-only operation
# Use sockets-based Level Zero IPC, avoid MPI/OFI entirely
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_TRANSPORT=mpi
export CCL_MNIC=local
export CCL_ALLREDUCE=ring
export CCL_PROCESS_LAUNCHER=none
export CCL_LOCAL_SIZE=8
export CCL_LOCAL_RANK=$SLURM_LOCALID
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_TOPOLIB=level_zero

echo "🔧 Distributed Training Configuration:"
echo "   - WORLD_SIZE: $WORLD_SIZE"
echo "   - MASTER_ADDR: $MASTER_ADDR"
echo "   - MASTER_PORT: $MASTER_PORT"
echo "   - CCL_ATL_TRANSPORT: $CCL_ATL_TRANSPORT"
echo "   - I_MPI_ROOT: $I_MPI_ROOT"
echo "   - CCL using Intel MPI with Level Zero IPC"
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