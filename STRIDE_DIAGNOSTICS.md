# Stride Mismatch Diagnostic Logging

## Overview
This document describes the diagnostic logging added to `engine/train.py` to identify the source of the PyTorch DDP stride mismatch warnings:

```
[W1017 12:09:57.211698780 reducer.cpp:327] Warning: Grad strides do not match bucket view strides
grad.sizes() = [64, 1, 3, 3], strides() = [9, 1, 3, 1]
bucket_view.sizes() = [64, 1, 3, 3], strides() = [9, 9, 3, 1]
```

## What Was Added

### 1. Gradient Hooks on All Parameters (Lines 511-538)
- **Purpose**: Automatically detect non-contiguous gradients after each backward pass
- **Location**: Registered after DDP wrapping
- **Output Format**:
  ```
  [STRIDE-DIAG][rank X] NON-CONTIGUOUS gradient in model_name.param_name:
    shape=(64, 1, 3, 3), stride=(9, 1, 3, 1)
    expected_stride=(9, 9, 3, 1)
    dtype=torch.float32, device=cuda:0
  ```
- **What to look for**: Any parameters that show non-contiguous gradients

### 2. Initial Parameter Memory Layout Check (Lines 540-554)
- **Purpose**: Verify all model parameters are contiguous at initialization
- **Timing**: Once at model setup, before training starts
- **Output Format**:
  ```
  [STRIDE-DIAG] Checking initial parameter memory layout:
    gen_diffusive_1: All parameters are contiguous ✓
    [WARNING] gen_diffusive_2.module.conv.weight is NON-CONTIGUOUS at init:
      shape=(64, 1, 3, 3), stride=(9, 1, 3, 1)
  ```
- **What to look for**: Any parameters that are already non-contiguous before training

### 3. Attention Conv Layer Properties (Lines 468-472)
- **Purpose**: Log the att_conv layer configuration
- **Output Format**:
  ```
  [STRIDE-DIAG] att_conv layer created:
    weight shape: (1, 512, 1, 1), stride: (512, 1, 1, 1)
    weight is_contiguous: True
  ```

### 4. Per-Iteration Diagnostics (Iteration 0 Only)
These logs run only on the first iteration to avoid spam:

#### a. Input Tensor Memory Format (Lines 707-710)
- **Purpose**: Check if channels_last conversion is working correctly
- **Location**: At start of D step
- **Output Format**:
  ```
  [STRIDE-DIAG][iter 0] Input tensor memory format (D step):
    cond_data1: is_channels_last=True
    real_data: is_channels_last=True
  ```

#### b. Discriminator Backward Pass (Lines 775-800)
- **Before backward**: Logs d_total properties
- **After backward**: Scans discriminator parameters for non-contiguous gradients
- **After optimizer step**: Confirms step completed

#### c. Generator Backward Pass (Lines 894-930)
- **Before backward**: Logs errG properties
- **After backward**: Scans both generator models for non-contiguous gradients
- **After optimizer step**: Confirms step completed

## How to Use This Information

### Step 1: Run Training and Collect Logs
```bash
# Your normal training command
python -m torch.distributed.launch --nproc_per_node=2 engine/train.py [args]
```

### Step 2: Search for Diagnostic Output
```bash
# Find all stride diagnostic messages
grep "STRIDE-DIAG" your_log_file.out

# Find non-contiguous gradients specifically
grep "NON-CONTIGUOUS" your_log_file.out
```

### Step 3: Interpret Results

#### If you see non-contiguous gradients at initialization:
This means the problem is in how the model is constructed. Look at the specific layer that's non-contiguous.

#### If gradients become non-contiguous during backward:
The gradient hook will show you exactly which parameter's gradient is non-contiguous. This tells you:
1. **Which model** (gen_diffusive_1, gen_diffusive_2, or disc_diffusive_2)
2. **Which layer** (e.g., module.conv1.weight)
3. **The exact stride pattern** causing the issue

#### Common Culprits to Investigate:

1. **Permute operations** in forward pass (e.g., in `up_or_down_sampling.py:131`)
2. **Channels_last memory format** interaction with certain layer types
3. **Custom operations** that create views rather than copies
4. **Gradient checkpointing** if enabled

## Expected Output Location

All logs will appear in your `.out` file mixed with regular training logs. They are prefixed with `[STRIDE-DIAG]` for easy filtering.

The first few iterations will have the most diagnostic output. After that, only the gradient hooks will fire if non-contiguous gradients are detected.

## Next Steps After Diagnosis

Once you identify which specific parameter has non-contiguous gradients:

1. **Trace the forward pass** for that layer to find view-creating operations
2. **Add `.contiguous()` calls** before operations that require contiguous tensors
3. **Consider disabling** `gradient_as_bucket_view=True` in DDP kwargs as a workaround (less memory efficient)
4. **Check if channels_last** is compatible with all your custom operations

## Notes

- Logs are rank 0 only (except gradient hooks which run on all ranks)
- Iteration 0 has the most detailed logging to minimize log spam
- The gradient hooks will catch non-contiguous gradients on ANY iteration
- This logging has minimal performance impact
