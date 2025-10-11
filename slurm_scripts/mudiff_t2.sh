#!/usr/bin/env bash
#
#SBATCH --job-name=MUDIFF_T2
#SBATCH --time=3-00:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # request 2 GPUs (required for MU-Diff distributed training)
#SBATCH --constraint=dgx           # DGX node
#SBATCH --error=log_mudiff_t2.%J.err
#SBATCH --output=log_mudiff_t2.%J.out

# ===============================
# User-configurable paths
# ===============================
# Experiment name (must match exp_name in YAML)
EXP_NAME="synthesize_T2"
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
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "====================================="
echo


module purge
module load miniconda
source activate mudiff

# ===============================
# Fix CUDA extension compilation issues
# ===============================

# Set CUDA paths
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# For A100:
export TORCH_CUDA_ARCH_LIST="8.0"

# Set a consistent PyTorch extensions cache directory (accessible to all processes)
# This prevents race conditions during JIT compilation in multi-process training
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "TORCH_EXTENSIONS_DIR set to: $TORCH_EXTENSIONS_DIR"

# Sanity check
python - <<'PY'
import os, pathlib
home=os.environ.get("CUDA_HOME")
print("CUDA_HOME=", home)
cands=[pathlib.Path(home)/"include"/"cuda_runtime.h",
       pathlib.Path(home)/"targets/x86_64-linux/include"/"cuda_runtime.h"]
print("cuda_runtime.h exists:", any(p.exists() for p in cands))
import shutil; print("nvcc:", shutil.which("nvcc"))
PY

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

# ===============================
# Pre-build CUDA extensions
# ===============================
echo
echo "====================================="
echo "Pre-building CUDA extensions..."
echo "====================================="

BUILD_SCRIPT="$REPO_ROOT/build_extensions.py"
if [ -f "$BUILD_SCRIPT" ]; then
    python "$BUILD_SCRIPT"
    BUILD_EXIT_CODE=$?

    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        echo
        echo "========================================="
        echo "WARNING: CUDA extension build failed!"
        echo "========================================="
        echo "The training might fail or fall back to slower CPU implementations."
        echo
        echo "To fix this issue, run the following command manually:"
        echo "  conda activate mudiff"
        echo "  conda install -c conda-forge libxcrypt"
        echo "  python $BUILD_SCRIPT"
        echo
        # Don't exit here, let training attempt to continue with fallback
    fi
else
    echo "WARNING: build_extensions.py not found at $BUILD_SCRIPT"
    echo "Skipping pre-build step. Extensions will be JIT compiled (may cause issues)."
fi

echo
echo "====================================="
echo "Starting training..."
echo "====================================="

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