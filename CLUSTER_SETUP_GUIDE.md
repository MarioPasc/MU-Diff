# MU-Diff Cluster Setup and Troubleshooting Guide

## Problem Summary

When running MU-Diff on a SLURM cluster with multi-GPU distributed training, the following error occurred:

```
fatal error: crypt.h: No such file or directory
RuntimeError: Error building extension 'fused'
ImportError: /localscratch/.../fused.so: cannot open shared object file
```

### Root Causes

1. **Missing `crypt.h` header**: Python 3.8's `Python.h` includes `crypt.h`, which is missing in the conda environment
2. **Multi-process JIT compilation race**: Two GPU processes trying to compile CUDA extensions simultaneously
3. **Extension cache conflicts**: Processes unable to properly share compiled `.so` files

---

## Solution: Complete Setup Guide

### Step 1: Install Missing Dependencies (ONE-TIME)

Before running any jobs, install the missing library in your conda environment:

```bash
# On the cluster login node or in an interactive session
conda activate mudiff
conda install -c conda-forge libxcrypt
```

**Why this is needed**: Python 3.8 requires `libxcrypt` for the `crypt.h` header file. Without it, C++ extensions cannot compile.

### Step 2: Verify the Installation

Check that the installation was successful:

```bash
conda activate mudiff
python -c "import torch; print(torch.__version__)"
python -c "import torch.utils.cpp_extension; print('OK')"
```

### Step 3: Pre-Build CUDA Extensions (RECOMMENDED)

The updated SLURM scripts will automatically attempt to pre-build the CUDA extensions. However, you can also do this manually before submitting jobs:

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MU-Diff
conda activate mudiff

# Set environment variables
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100 GPUs
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"

# Build extensions
python build_extensions.py
```

**Expected output**:
```
Building fused extension...
[SUCCESS] fused extension built successfully!
Building upfirdn2d extension...
[SUCCESS] upfirdn2d extension built successfully!
✓ All extensions built successfully!
```

### Step 4: Submit Your SLURM Job

Now you can submit your job as usual:

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MU-Diff
sbatch slurm_scripts/mudiff_flair.sh
```

The script will:
1. Activate the mudiff conda environment
2. Set up CUDA paths and extensions directory
3. Verify configuration and dataset
4. **Automatically pre-build CUDA extensions**
5. Start distributed training

---

## What Was Fixed

### Code-Level Changes

#### 1. **utils/op/fused_act.py**
- Added try-except block around CUDA extension loading
- Added proper error handling with fallback message
- Set `build_directory` parameter for consistent cache location
- Extensions now check if compilation succeeded before use

#### 2. **utils/op/upfirdn2d.py**
- Same improvements as fused_act.py
- Better error messages
- Graceful fallback handling

#### 3. **build_extensions.py** (NEW FILE)
- Pre-compilation script for CUDA extensions
- Compiles `fused` and `upfirdn2d` extensions before training starts
- Comprehensive error reporting
- Prevents multi-process compilation conflicts

### Cluster-Level Changes

#### All SLURM Scripts (mudiff_flair.sh, mudiff_t2.sh, mudiff_t1.sh, mudiff_t1ce.sh)

**Added environment setup:**
```bash
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
```

**Added pre-build section:**
- Automatically runs `build_extensions.py` before training
- Provides helpful error messages if build fails
- Allows training to continue with fallback (won't crash immediately)

---

## Troubleshooting

### Issue: Build still fails with crypt.h error

**Solution**: Make sure libxcrypt is installed in the correct environment:
```bash
conda activate mudiff
conda list libxcrypt  # Should show the package
conda install -c conda-forge libxcrypt --force-reinstall
```

### Issue: Permission denied on TORCH_EXTENSIONS_DIR

**Solution**: The directory is set to `$REPO_ROOT/.torch_extensions`. Make sure you have write permissions:
```bash
chmod 755 /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MU-Diff/.torch_extensions
```

### Issue: nvcc not found

**Solution**: Check CUDA installation:
```bash
conda activate mudiff
which nvcc
echo $CUDA_HOME
```

If nvcc is not found, install it:
```bash
conda install -c nvidia cuda-toolkit
```

### Issue: Extensions compile but training still crashes

**Possible causes**:
1. Different CUDA architectures - make sure `TORCH_CUDA_ARCH_LIST="8.0"` matches your GPU (A100 = 8.0)
2. Out of memory - reduce batch size or use `--use_grad_checkpoint`
3. Wrong PyTorch/CUDA version mismatch

Check versions:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Issue: "ninja: build stopped: subcommand failed"

**Solution**: This is often a compiler incompatibility. Check your GCC version:
```bash
gcc --version  # Should be 7.x - 11.x for CUDA 12.1
```

If needed, specify a different compiler:
```bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

---

## Performance Notes

### With Pre-Built Extensions (RECOMMENDED)
- ✅ Fast startup (extensions already compiled)
- ✅ No multi-process conflicts
- ✅ Consistent performance across runs
- ✅ Better error handling

### With JIT Compilation (OLD BEHAVIOR)
- ❌ Slow first run (compilation takes 2-5 minutes)
- ❌ Possible race conditions with multi-GPU
- ❌ Cache conflicts
- ❌ Cryptic error messages

---

## Architecture Overview

For reference, here's how MU-Diff components interact:

```
Training Script (train.py)
    ↓
Multi-GPU Spawn (DDP)
    ↓
├─ Process 1 (GPU 0) ───┐
│   ├─ Import backbones  │
│   │   └─ Import up_or_down_sampling
│   │       └─ Import utils/op (CUDA extensions)
│   ├─ Generator 1       │
│   ├─ Generator 2       │
│   └─ Discriminator     │
│                        │
└─ Process 2 (GPU 1) ───┘
    ├─ Import backbones
    │   └─ Import up_or_down_sampling
    │       └─ Import utils/op (CUDA extensions)
    ├─ Generator 1
    ├─ Generator 2
    └─ Discriminator

Both processes MUST share the same compiled extensions!
```

**Key**: The CUDA extensions (fused_bias_act, upfirdn2d) must be compiled BEFORE multiprocessing starts, otherwise both processes try to compile simultaneously and fail.

---

## Quick Reference Commands

```bash
# Check environment
conda activate mudiff
python -c "import torch; print(torch.cuda.is_available())"

# Manual pre-build
cd /path/to/MU-Diff
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
python build_extensions.py

# Submit job
sbatch slurm_scripts/mudiff_flair.sh

# Check job status
squeue -u $USER
tail -f log_mudiff_FLAIR.*.out

# If job fails, check error log
tail -100 log_mudiff_FLAIR.*.err
```

---

## Files Modified

### Code Files
- `utils/op/fused_act.py` - Better error handling, cache directory support
- `utils/op/upfirdn2d.py` - Better error handling, cache directory support
- `build_extensions.py` - NEW: Pre-compilation script

### SLURM Scripts
- `slurm_scripts/mudiff_flair.sh` - Added TORCH_EXTENSIONS_DIR, pre-build step
- `slurm_scripts/mudiff_t2.sh` - Added TORCH_EXTENSIONS_DIR, pre-build step
- `slurm_scripts/mudiff_t1.sh` - Added TORCH_EXTENSIONS_DIR, pre-build step
- `slurm_scripts/mudiff_t1ce.sh` - Added TORCH_EXTENSIONS_DIR, pre-build step

---

## Support

If you continue to have issues after following this guide:

1. Check the error log: `log_mudiff_*.err`
2. Verify all environment variables are set correctly
3. Try running `build_extensions.py` manually to see detailed error messages
4. Check that your CUDA version matches PyTorch's CUDA version

Generated: 2025-10-11
