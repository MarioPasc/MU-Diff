#!/usr/bin/env bash
#
#SBATCH --job-name=MUDIFF_T2
#SBATCH --time=3-00:00:00          # 3 days
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # request 2 GPUs (required for MU-Diff distributed training)
#SBATCH --constraint=dgx           # DGX node
#SBATCH --error=log_mudiff_t2.%J.err
#SBATCH --output=log_mudiff_t2.%J.out

# Print job information
echo "====================================="
echo "SLURM JOB: T2 Synthesis with MU-Diff"
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

# Set up environment
echo "Setting up environment..."
module purge
module load cuda/11.8
module load python/3.9

# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Check Python and CUDA installation
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"
echo

# Set CUDA environment variables for 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH

# Set multiprocessing method for PyTorch distributed training
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

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

# Navigate to experiment directory
cd experiments

# Verify configuration file exists
if [ ! -f "cfg/local.yaml" ]; then
    echo "ERROR: Configuration file cfg/local.yaml not found!"
    exit 1
fi

# Verify dataset exists
echo "Checking dataset availability..."
if [ ! -d "../data" ]; then
    echo "ERROR: Dataset directory '../data' not found!"
    echo "Please ensure the preprocessed dataset is uploaded to the supercomputer."
    exit 1
fi

# Run the complete T2 synthesis experiment (train + test)
echo "Starting T2 synthesis experiment..."
echo "Command: python run.py -c cfg/local.yaml -e synthesize_T2"
echo "This will:"
echo "  1. Train the MU-Diff model for T2 synthesis"
echo "  2. Log training metrics and validation scores"
echo "  3. Perform testing and log test metrics"
echo "  4. Save model checkpoints and generated samples"
echo

# Execute the experiment
python run.py -c cfg/local.yaml -e synthesize_T2

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "====================================="
    echo "T2 synthesis experiment completed successfully!"
    echo "====================================="
    echo "Results saved in: ../results/synthesize_T2/"
    echo "Check the following outputs:"
    echo "  - Model checkpoints: ../results/synthesize_T2/gen_diffusive_*.pth"
    echo "  - Training samples: ../results/synthesize_T2/sample_discrete_epoch_*.png"
    echo "  - Validation metrics: ../results/synthesize_T2/val_*.npy"
    echo "  - Test results: ../results/synthesize_T2/generated_samples/"
else
    echo
    echo "====================================="
    echo "ERROR: T2 synthesis experiment failed!"
    echo "====================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error logs: log_mudiff_t2.$SLURM_JOB_ID.err"
    exit $EXIT_CODE
fi

echo "End Time: $(date)"
echo "Total GPU time: $(($(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s))) seconds"
