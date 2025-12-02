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
module load GCCcore/14.2.0 libfabric/2.0.0
module load WebProxy  # Required for internet access (HuggingFace, WandB)

# Explicitly set I_MPI_ROOT if not already set by the module
if [ -z "$I_MPI_ROOT" ]; then
    # Try to find Intel MPI installation from loaded modules
    # EasyBuild (used by many HPC clusters) sets EBROOTIIMPI
    if [ -n "$EBROOTIIMPI" ]; then
        export I_MPI_ROOT=$EBROOTIIMPI/mpi
        echo "Setting I_MPI_ROOT from EasyBuild: $I_MPI_ROOT"
    elif [ -n "$EBROOTIMPI" ]; then
        export I_MPI_ROOT=$EBROOTIMPI
        echo "Setting I_MPI_ROOT from EasyBuild: $I_MPI_ROOT"
    elif [ -n "$MPIROOT" ]; then
        export I_MPI_ROOT=$MPIROOT
        echo "Setting I_MPI_ROOT from MPIROOT: $I_MPI_ROOT"
    elif [ -n "$MPI_ROOT" ]; then
        export I_MPI_ROOT=$MPI_ROOT
        echo "Setting I_MPI_ROOT from MPI_ROOT: $I_MPI_ROOT"
    else
        echo "Warning: Could not determine I_MPI_ROOT from environment"
        echo "Available environment variables:"
        env | grep -i mpi | head -20
        
        # Common installation paths - adjust based on your cluster
        for path in /opt/intel/oneapi/mpi/latest /usr/local/intel/mpi /opt/intel/mpi; do
            if [ -d "$path" ]; then
                export I_MPI_ROOT=$path
                echo "Found Intel MPI at: $I_MPI_ROOT"
                break
            fi
        done
    fi
fi

# If we found I_MPI_ROOT, also set ONEAPI_ROOT for compatibility
if [ -n "$I_MPI_ROOT" ] && [ -z "$ONEAPI_ROOT" ]; then
    # Try to derive ONEAPI_ROOT from I_MPI_ROOT
    ONEAPI_CANDIDATE=$(dirname "$I_MPI_ROOT")
    if [ -d "$ONEAPI_CANDIDATE" ]; then
        export ONEAPI_ROOT=$ONEAPI_CANDIDATE
        echo "Setting ONEAPI_ROOT to: $ONEAPI_ROOT"
    fi
fi

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
export USE_XETLA=OFF
export SYCL_CACHE_PERSISTENT=1
export FI_PROVIDER=tcp
export FI_TCP_IFACE=lo
# export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export CCL_ATL_TRANSPORT=ofi
# export CCL_ZE_IPC_EXCHANGE=sockets
# export CCL_MNIC=local
export CCL_LOCAL_SIZE=8
export CCL_LOCAL_RANK=$SLURM_LOCALID
export CCL_ATL_SHM=1
export CCL_LOG_LEVEL=info
# export CCL_SAME_STREAM=1
# export CCL_BLOCKING_WAIT=0
# export CCL_ALLREDUCE=ring
export CCL_PROCESS_LAUNCHER=none
# export I_MPI_OFFLOAD=1
# export I_MPI_OFFLOAD_TOPOLIB=level_zero

echo "🔧 Distributed Training Configuration:"
echo "   - WORLD_SIZE: $WORLD_SIZE"
echo "   - MASTER_ADDR: $MASTER_ADDR"
echo "   - MASTER_PORT: $MASTER_PORT"
echo "   - SLURM assigned CPUs: $SLURM_CPUS"
echo "   - First CPU extracted: $FIRST_CPU"
echo "   - CCL_WORKER_COUNT: $CCL_WORKER_COUNT"
echo "   - CCL_WORKER_AFFINITY: $CCL_WORKER_AFFINITY"
echo "   - CCL_ATL_TRANSPORT: $CCL_ATL_TRANSPORT"
echo "   - I_MPI_ROOT: ${I_MPI_ROOT:-NOT SET}"
echo "   - I_MPI_OFFLOAD: $I_MPI_OFFLOAD"
echo "   - FI_INFO: $(fi_info)"
echo "   - FI_PROVIDER: $FI_PROVIDER"
echo "   - FI_TCP_IFACE: $FI_TCP_IFACE"
echo ""

# Verify Intel MPI is properly configured
if [ -z "$I_MPI_ROOT" ]; then
    echo "⚠️  WARNING: I_MPI_ROOT is not set! oneCCL may fail to initialize."
    echo "⚠️  Loaded modules:"
    module list
else
    echo "✅ Intel MPI configured at: $I_MPI_ROOT"
fi
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