#!/usr/bin/env bash
#
# Master script to submit all MU-Diff experiments to SLURM
# This script submits one job per experiment/model as required
#

echo "========================================"
echo "MU-Diff SLURM Job Submission Script"
echo "========================================"
echo "This script will submit 4 separate SLURM jobs:"
echo "  1. T1CE synthesis experiment"
echo "  2. FLAIR synthesis experiment" 
echo "  3. T2 synthesis experiment"
echo "  4. T1 synthesis experiment"
echo ""
echo "Each job will:"
echo "  - Use 2 GPUs (required for MU-Diff distributed training)"
echo "  - Run for up to 3 days"
echo "  - Use 250GB RAM and 8 CPU cores"
echo "  - Perform complete training + testing + logging"
echo ""

# Check if we're in the right directory
if [ ! -d "slurm_scripts" ]; then
    echo "ERROR: slurm_scripts directory not found!"
    echo "Please run this script from the MU-Diff root directory."
    exit 1
fi

# Check if SLURM scripts exist
for script in mudiff_t1ce.sh mudiff_flair.sh mudiff_t2.sh mudiff_t1.sh; do
    if [ ! -f "slurm_scripts/$script" ]; then
        echo "ERROR: SLURM script slurm_scripts/$script not found!"
        exit 1
    fi
done

echo "All SLURM scripts found. Proceeding with job submission..."
echo ""

# Submit jobs
echo "Submitting T1CE synthesis job..."
JOB1=$(sbatch slurm_scripts/mudiff_t1ce.sh | grep -o '[0-9]\+')
echo "  Submitted job ID: $JOB1"

echo "Submitting FLAIR synthesis job..."
JOB2=$(sbatch slurm_scripts/mudiff_flair.sh | grep -o '[0-9]\+')
echo "  Submitted job ID: $JOB2"

echo "Submitting T2 synthesis job..."
JOB3=$(sbatch slurm_scripts/mudiff_t2.sh | grep -o '[0-9]\+')
echo "  Submitted job ID: $JOB3"

echo "Submitting T1 synthesis job..."
JOB4=$(sbatch slurm_scripts/mudiff_t1.sh | grep -o '[0-9]\+')
echo "  Submitted job ID: $JOB4"

echo ""
echo "========================================"
echo "All jobs submitted successfully!"
echo "========================================"
echo "Job IDs:"
echo "  T1CE:  $JOB1"
echo "  FLAIR: $JOB2"
echo "  T2:    $JOB3"
echo "  T1:    $JOB4"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1,$JOB2,$JOB3,$JOB4"
echo ""
echo "Check job status with:"
echo "  scontrol show job <job_id>"
echo ""
echo "View logs in real-time with:"
echo "  tail -f log_mudiff_*.out"
echo "  tail -f log_mudiff_*.err"
echo ""
echo "Cancel jobs if needed with:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4"
echo ""
echo "Expected runtime: 2-3 days per job"
echo "Results will be saved in: ./results/synthesize_<modality>/"
