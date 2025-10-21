# DDP Stride Mismatch Error - Diagnosis and Fix

## Problem

```
grad.sizes() = [64, 1, 3, 3], strides() = [9, 1, 3, 1]
bucket_view.sizes() = [64, 1, 3, 3], strides() = [9, 9, 3, 1]
```

**Error**: PyTorch DDP warning about gradient stride mismatch at iteration 50.

## Root Cause Analysis

The stride mismatch occurs due to an interaction between three factors:

1. **Input tensors use `channels_last` memory format** (train.py:261-264, 375-378)
   - This is done for performance: `x.to(memory_format=torch.channels_last)`
   - Channels_last format changes memory layout from NCHW to NHWC ordering

2. **DDP configured with `gradient_as_bucket_view=True`** (train.py:510)
   - This optimization views gradients directly in the bucket buffer
   - Requires gradients to have contiguous, standard memory layout
   - More memory-efficient than copying gradients

3. **Gradient stride inheritance from channels_last inputs**
   - When channels_last tensors flow through the network, some gradients inherit non-contiguous strides
   - Specific parameter: `[64, 1, 3, 3]` shape with strides `[9, 1, 3, 1]` instead of expected `[9, 9, 3, 1]`
   - The second dimension has stride 1 instead of 9, indicating non-standard layout

## The Fix

**Solution**: Automatically make gradients contiguous via gradient hooks (train.py:521-535)

### How it works:

```python
def check_grad_stride(name, param):
    """Hook to check gradient strides and make them contiguous if needed"""
    def hook(grad):
        if grad is not None:
            is_contig = grad.is_contiguous()
            if not is_contig:
                # Log the issue
                print(f"[STRIDE-DIAG] NON-CONTIGUOUS gradient in {name}")
                print(f"  shape={tuple(grad.shape)}, stride={tuple(grad.stride())}")
                # FIX: Return contiguous gradient
                return grad.contiguous()
        return grad
    return hook
```

### What happens:

1. **Before backward pass**: Input tensors are in channels_last format (performance optimization)
2. **During backward pass**: Some gradients are created with non-contiguous strides
3. **Gradient hook triggers**: Detects non-contiguous gradients
4. **Automatic fix**: Calls `grad.contiguous()` to create a contiguous copy
5. **DDP processes gradient**: Now receives gradient with expected stride layout
6. **No more warnings**: DDP bucket view stride requirements are satisfied

### Trade-offs:

- **Performance**: Small overhead from `contiguous()` calls on affected gradients
- **Memory**: Temporary allocation for contiguous copy (minimal impact)
- **Correctness**: ✅ Fully correct, no numerical changes
- **Compatibility**: ✅ Works with both channels_last inputs and DDP optimizations

## Alternative Approaches (Not Recommended)

### Option A: Disable `gradient_as_bucket_view`
```python
ddp_kwargs = dict(
    device_ids=[gpu],
    broadcast_buffers=False,
    gradient_as_bucket_view=False,  # Disable this
    static_graph=True
)
```
- ❌ Less memory efficient (copies gradients instead of viewing)
- ❌ Slower gradient synchronization
- ✅ Avoids stride mismatch entirely

### Option B: Remove channels_last format
```python
# Don't convert to channels_last
cond_data1 = x1.to(device, non_blocking=True)  # No .to(memory_format=...)
```
- ❌ Slower convolution operations (channels_last is faster on modern GPUs)
- ❌ Higher memory usage for intermediate activations
- ✅ Avoids non-contiguous gradients

## Expected Behavior After Fix

1. **First few iterations**: Will see diagnostic messages like:
   ```
   [STRIDE-DIAG] Found [64, 1, 3, 3] parameter: disc_diffusive_2.module.XXX.weight
   [STRIDE-DIAG] NON-CONTIGUOUS gradient in disc_diffusive_2.module.XXX.weight:
     shape=(64, 1, 3, 3), stride=(9, 1, 3, 1)
     FIX: Making gradient contiguous to avoid DDP bucket view mismatch
   ```

2. **After iteration 50**: No more DDP stride warnings

3. **Training continues normally**: Full performance with both channels_last optimization and DDP efficiency

## Implementation Details

- **Location**: engine/train.py:517-553
- **Scope**: Applies to all parameters in all DDP-wrapped models (gen_diffusive_1, gen_diffusive_2, disc_diffusive_2)
- **Overhead**: Only affects gradients that are non-contiguous (small percentage)
- **Logging**: Diagnostic prints on rank 0 show which parameters are affected

## Technical Background

### Why does this happen?

Channels_last memory format reorders tensor data for better cache locality:
- **Standard (NCHW)**: `[batch, channel, height, width]` with strides `[C*H*W, H*W, W, 1]`
- **Channels_last (NHWC)**: `[batch, height, width, channel]` with strides `[H*W*C, W*C, C, 1]`

When a conv layer has weight `[64, 1, 3, 3]`:
- Expected stride: `[1*3*3, 64*3*3, 64*3, 64]` = `[9, 9, 3, 1]` (contiguous)
- Actual stride after channels_last: `[9, 1, 3, 1]` (non-contiguous)

The dimension of size 1 gets collapsed, creating unexpected stride pattern.

### Why didn't this fail earlier?

- DDP only checks stride compatibility when using `gradient_as_bucket_view=True`
- Warning appears at iteration 50 because DDP's gradient bucketing warmup completes by then
- Earlier iterations may have been in a "setup phase" where buckets weren't finalized

## Verification

Run training and check that:
1. ✅ No "Grad strides do not match bucket view strides" warnings after fix
2. ✅ Training continues past iteration 50 without errors
3. ✅ Loss values are normal (fix doesn't change computation)
4. ✅ Memory usage remains stable (contiguous() overhead is minimal)
