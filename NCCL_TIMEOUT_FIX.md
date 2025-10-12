# NCCL Timeout Fix for MU-Diff Distributed Training

## Problem Summary

The MU-Diff training was experiencing NCCL timeout errors during distributed training on supercomputing clusters:

```
[Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=668, OpType=ALLREDUCE,
NumelIn=52, NumelOut=52, Timeout(ms)=600000) ran for 600097 milliseconds before timing out.
```

### Root Causes Identified

1. **NCCL Timeout Too Short**: Default 10-minute timeout was insufficient for supercomputing clusters where operations may take longer due to shared resources
2. **Data Loader Timeout Misconfiguration**: 600-second (10 min) timeout matched NCCL timeout, causing synchronization issues when data loaders hung
3. **Insufficient Error Handling**: No try-catch blocks to gracefully handle data loader failures
4. **File Descriptor Exhaustion**: Too many data loader workers (8 per rank × 2 ranks = 16 workers) led to "too many open files" errors
5. **Redundant Parameter Broadcasting**: `broadcast_params()` called before DDP wrapping was unnecessary and could cause sync issues
6. **RETAIN_GRAPH Flag**: Set to `True` unnecessarily, potentially causing memory issues

## Changes Implemented

### 1. NCCL Timeout Configuration (lines 12-21)

**Before:**
```python
os.environ.setdefault("NCCL_DEBUG", "WARN")
```

**After:**
```python
# NCCL timeout configuration for distributed training
nccl_timeout_minutes = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "30"))
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

import datetime
torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT = datetime.timedelta(minutes=nccl_timeout_minutes)
```

**Benefits:**
- Increased timeout from 10 to 30 minutes (configurable via environment variable)
- Enabled async error handling for better debugging
- Enabled blocking wait to catch errors immediately

### 2. Data Loader Configuration (lines 412-419)

**Before:**
```python
tw = int(os.environ.get("MU_TRAIN_WORKERS", "8"))
vw = int(os.environ.get("MU_VAL_WORKERS", "4"))
timeout_s = int(os.environ.get("MU_DL_TIMEOUT", "600"))
```

**After:**
```python
# Reduced default workers from 8 to 4 to avoid file descriptor exhaustion
tw = int(os.environ.get("MU_TRAIN_WORKERS", "4"))
vw = int(os.environ.get("MU_VAL_WORKERS", "2"))
# Reduced timeout from 600s to 120s to catch hangs earlier
timeout_s = int(os.environ.get("MU_DL_TIMEOUT", "120"))
```

**Benefits:**
- Fewer workers reduce file descriptor usage and prevent "too many open files" errors
- Shorter timeout (2 minutes) catches data loader issues before NCCL timeout
- Still configurable via environment variables if more workers are needed

### 3. Improved Error Handling in init_processes (lines 986-1042)

**Before:**
```python
dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
try:
    fn(rank, gpu, args)
    dist.barrier()
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
```

**After:**
```python
timeout = datetime.timedelta(minutes=nccl_timeout_minutes)
dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size, timeout=timeout)

training_succeeded = False
try:
    fn(rank, gpu, args)
    training_succeeded = True
    if rank == 0:
        print(f"[rank: {rank}] Training completed successfully, synchronizing ranks...", flush=True)
    dist.barrier()
except Exception as e:
    print(f"[rank: {rank}] ERROR during training: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise
finally:
    if dist.is_initialized():
        try:
            if not training_succeeded:
                time.sleep(2)  # Give other ranks time to detect error
            dist.destroy_process_group()
        except Exception as e:
            print(f"[rank: {rank}] Error destroying process group: {e}", flush=True)
```

**Benefits:**
- Explicit timeout passed to init_process_group
- Only calls barrier if training succeeded (prevents hanging on errors)
- Proper exception handling and traceback printing
- Graceful cleanup even on errors

### 4. Removed Redundant broadcast_params (line 468-469)

**Before:**
```python
broadcast_params(gen_diffusive_1.parameters())
broadcast_params(gen_diffusive_2.parameters())
broadcast_params(disc_diffusive_2.parameters())
```

**After:**
```python
# Note: broadcast_params removed - DDP automatically synchronizes parameters during initialization
# This avoids redundant communication and potential synchronization issues
```

**Benefits:**
- Eliminates redundant parameter synchronization
- DDP handles this automatically during initialization
- Reduces potential synchronization points where hangs can occur

### 5. Fixed RETAIN_GRAPH Flag (line 25)

**Before:**
```python
RETAIN_GRAPH: bool = True
```

**After:**
```python
RETAIN_GRAPH: bool = False
```

**Benefits:**
- Proper memory cleanup after backward passes
- Prevents potential memory leaks and graph accumulation
- Should only be True for debugging purposes

### 6. Added Configuration Logging (lines 520-528)

```python
print(f"\n[CONFIG] Distributed Training Configuration:", flush=True)
print(f"  - World size: {args.world_size}", flush=True)
print(f"  - NCCL timeout: {nccl_timeout_minutes} minutes", flush=True)
print(f"  - Data loader workers (train/val): {tw}/{vw}", flush=True)
print(f"  - Data loader timeout: {timeout_s} seconds", flush=True)
```

**Benefits:**
- Easy verification of configuration at training start
- Helps debugging by showing actual values being used

### 7. Added Heartbeat Logging (lines 866-868)

```python
if iteration > 0 and iteration % 50 == 0:
    print(f"[rank {rank}] Heartbeat: epoch={epoch} iter={iteration} global_step={global_step}", flush=True)
```

**Benefits:**
- Helps detect which rank is hanging or progressing slower
- Provides regular progress updates from all ranks

## How to Use

### Default Configuration (Recommended)
Just run your training as before. The new defaults should work for most cases:
```bash
python engine/train.py [your arguments]
```

### Custom NCCL Timeout
If you need longer timeout (e.g., for very slow storage):
```bash
export NCCL_TIMEOUT_MINUTES=60  # 1 hour timeout
python engine/train.py [your arguments]
```

### Custom Data Loader Configuration
If you have fast storage and want more workers:
```bash
export MU_TRAIN_WORKERS=8
export MU_VAL_WORKERS=4
export MU_DL_TIMEOUT=300  # 5 minutes
python engine/train.py [your arguments]
```

### Debugging NCCL Issues
Enable verbose NCCL logging:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
python engine/train.py [your arguments]
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_TIMEOUT_MINUTES` | 30 | NCCL operation timeout in minutes |
| `MU_TRAIN_WORKERS` | 4 | Number of data loader workers for training |
| `MU_VAL_WORKERS` | 2 | Number of data loader workers for validation |
| `MU_DL_TIMEOUT` | 120 | Data loader timeout in seconds |
| `MU_PREFETCH` | 2 | Data loader prefetch factor |
| `MU_PERSISTENT` | 0 | Whether to use persistent workers (1=yes, 0=no) |
| `NCCL_DEBUG` | WARN | NCCL logging level (WARN, INFO, TRACE) |

## Testing Recommendations

1. **Test with small dataset first**: Verify the training loop works without timeouts
2. **Monitor all ranks**: Check that heartbeat messages appear from all ranks
3. **Check file descriptors**: Run `lsof -p <pid> | wc -l` to verify worker count doesn't exhaust FDs
4. **Gradual scaling**: Start with 2 GPUs, then scale up to more

## Expected Behavior

### Successful Training
You should see:
- Configuration printed at start
- Heartbeat messages from all ranks every 50 iterations
- Training progresses without hangs
- Clean shutdown with "Training completed successfully" message

### Error Scenarios
If a rank fails, you should now see:
- Clear error message with traceback from the failing rank
- Other ranks detect the error quickly (within 2 minutes instead of 10)
- Proper cleanup and process group destruction

## Additional Notes

- The NCCL timeout increase doesn't slow down training - it only affects how long to wait before declaring a deadlock
- Reducing workers may slightly slow data loading but improves stability
- The heartbeat messages add negligible overhead
- All changes are backward compatible with existing configurations

## Troubleshooting

### If timeouts still occur:
1. Increase `NCCL_TIMEOUT_MINUTES` further
2. Check network connectivity between nodes
3. Monitor GPU utilization to ensure all GPUs are working
4. Check system logs for hardware issues

### If data loading is slow:
1. Increase `MU_TRAIN_WORKERS` (but monitor file descriptors)
2. Increase `MU_DL_TIMEOUT`
3. Consider using `MU_PERSISTENT=1` for persistent workers

### If seeing "too many open files":
1. Reduce `MU_TRAIN_WORKERS` and `MU_VAL_WORKERS`
2. Increase system ulimit: `ulimit -n 65536`
3. Check cluster-specific file descriptor limits
