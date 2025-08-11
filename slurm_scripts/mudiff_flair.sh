#!/usr/bin/env bash
#
#SBATCH --job-name=MUDIFF_FLAIR
#SBATCH --time=3-00:00:00          # 3 days
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # request 2 GPUs (required for MU-Diff distributed training)
#SBATCH --constraint=dgx           # DGX node
#SBATCH --error=log_mudiff_flair.%J.err
#SBATCH --output=log_mudiff_flair.%J.out

# ===============================
# User-configurable paths
# ===============================
# Experiment name (must match exp_name in YAML)
EXP_NAME="synthesize_FLAIR"
# User gives root path. 
ROOT="/mnt/home/users/tic_163_uma/mpascual/execs/MUDIFF"
# Key directories/files
CONFIG_FILE="$ROOT/cfg.yaml"
DATA_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/diffusion_brats"
# Local results root (for info only; actual output_root may be overridden by YAML)
RESULTS_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/results"
RESULTS_DIR="$RESULTS_ROOT/$EXP_NAME"

# Derive repo root from the run.py path and set PYTHONPATH accordingly
REPO_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MU-Diff"
RUN_PY="$REPO_ROOT/experiments/run.py"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
else
    echo "Results directory already exists: $RESULTS_DIR"
fi


# Print job information
echo "====================================="
echo "SLURM JOB: FLAIR Synthesis with MU-Diff"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "====================================="
echo

echo "Repository Root: $REPO_ROOT"

# Set up environment
echo "Setting up environment..."
source load mudiff

# Check Python and CUDA installation
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"
echo

# Set CUDA environment variables for 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
# Ensure Python can import project packages (backbones, dataset, etc.)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Set multiprocessing method for PyTorch distributed training
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="8.0"

# Check GPU availability
echo "Available GPUs:"
nvidia-smi
echo

# Check that we have exactly 2 GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ $GPU_COUNT -lt 2 ]; then
    echo "ERROR: MU-Diff requires at least 2 GPUs, but only $GPU_COUNT found!"
    exit 1
fi
echo "GPU count verified: $GPU_COUNT GPUs available"
echo

# Navigate to experiment directory (use absolute paths instead)
# cd experiments

# Verify configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Verify dataset exists
echo "Checking dataset availability..."
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Dataset directory '$DATA_DIR' not found!"
    echo "Please ensure the preprocessed dataset is uploaded to the supercomputer."
    exit 1
fi

# Execute the experiment
python "$RUN_PY" -c "$CONFIG_FILE" -e "$EXP_NAME"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "=================================================="
    echo "FLAIR synthesis experiment completed successfully!"
    echo "=================================================="
    echo "Results saved in (local default): $RESULTS_DIR/"
    echo "Note: If output_root is set in YAML, results are under that root:"
    echo "  - Model checkpoints: $RESULTS_DIR/gen_diffusive_*.pth"
    echo "  - Training samples: $RESULTS_DIR/sample_discrete_epoch_*.png"
    echo "  - Validation metrics: $RESULTS_DIR/val_*.npy"
    echo "  - Test results: $RESULTS_DIR/generated_samples/"
else
    echo
    echo "========================================="
    echo "ERROR: FLAIR synthesis experiment failed!"
    echo "========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error logs: log_mudiff_flair.$SLURM_JOB_ID.err"
    exit $EXIT_CODE
fi

echo "End Time: $(date)"
echo "Total GPU time: $(($(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s))) seconds"
