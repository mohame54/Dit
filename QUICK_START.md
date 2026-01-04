# Quick Start - Fixed FSDP Training

## What Was Fixed

### 🔴 The Error
```
RuntimeError: Found dtype Float but expected Half
```

### ✅ The Solution

**Root Cause**: Dtype mismatch in mixed precision training - model expected float16 but received float32 in some operations.

**Fixes Applied**:
1. ✅ Cast inputs to model dtype in training loop
2. ✅ Made positional encoding dtype-aware
3. ✅ Fixed time embedding to use correct dtype
4. ✅ Set gradient reductions to fp32 for stability
5. ✅ Fixed device assignment (LOCAL_RANK vs RANK)

---

## Quick Test

### Single Node (8 GPUs)
```bash
torchrun --nproc_per_node=8 main.py \
    --epochs 100 \
    --train-batch-sz 128 \
    --val-batch-sz 16 \
    --mp-dt float16 \
    --use-scheduler True \
    --scheduler-type cosine \
    --data-dir-path data \
    --dir-path-save checkpoints \
    --epoch-save-freq 10 \
    --save-best True
```

### Multi-Node (2 nodes, 8 GPUs each)
```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=<NODE0_IP> --master_port=29500 main.py <args>

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=<NODE0_IP> --master_port=29500 main.py <args>
```

---

## New Features Added

### 1. Learning Rate Schedulers
```bash
# Cosine annealing (recommended)
--use-scheduler True --scheduler-type cosine

# Step LR
--use-scheduler True --scheduler-type step --scheduler-step-size 100 --scheduler-gamma 0.5

# Disable
--use-scheduler False
```

### 2. Auto-Save Best Model
```bash
--save-best True  # Saves checkpoint when validation loss improves
```

### 3. Better Logging
- Shows current learning rate each epoch
- Tracks best validation loss
- More detailed epoch summaries

---

## Key Changes Summary

| File | Change | Impact |
|------|--------|--------|
| `model.py` | Dtype-aware positional encoding | **Fixes the main error** |
| `model.py` | Time embedding dtype fix | Ensures consistency |
| `train_utils.py` | Cast inputs to model dtype | **Critical for mixed precision** |
| `fsdp_utils.py` | Auto-wrap policy | Better memory efficiency |
| `fsdp_utils.py` | fp32 gradient reductions | More stable training |
| `main.py` | LOCAL_RANK for devices | **Fixes multi-node training** |
| `main.py` | LR scheduler support | Better convergence |
| `main.py` | Best model tracking | Convenience feature |

---

## Verify It Works

After running, you should see:
```
Started Training on: 1 / 100 epoch
Current LR: 1.000000e-04
Training: 100%|████████| 625/625 [02:34<00:00, 4.04it/s, loss=0.1234]
Validation: 100%|████████| 100/100 [00:15<00:00, 6.67it/s, val_loss=0.1145]
Epoch 1 completed in 169.23 seconds
Average Train Loss: 0.123400
Average Val Loss: 0.114500
>>>>>> New Best Val Loss: 0.114500
```

**No more dtype errors!** ✅

---

## Performance Tips

1. **Batch Size**: Start with 128, increase if GPU memory allows
2. **Mixed Precision**: Use float16 (or bfloat16 if supported)
3. **Gradient Accumulation**: If OOM, modify `grad_accum_steps` in `train_utils.py`
4. **Checkpoint Frequency**: Balance disk space vs. recovery points

---

## Troubleshooting

### Still Getting Dtype Errors?
- Check your data loader returns float32 tensors
- Verify `--mp-dt float16` is set
- Check PyTorch version (>=2.0 recommended)

### OOM (Out of Memory)?
- Reduce batch size
- Enable activation checkpointing (advanced)
- Use SHARD_GRAD_OP instead of FULL_SHARD

### Slow Training?
- Check all GPUs are being used (`nvidia-smi`)
- Verify data loading isn't bottleneck
- Consider increasing `num_workers` in data loader

---

## Next Steps

1. **Test**: Run for 1-2 epochs to verify everything works
2. **Monitor**: Watch GPU utilization and memory
3. **Scale**: If working, increase to full epochs
4. **Tune**: Adjust learning rate, scheduler, etc.

---

For detailed explanation of all changes, see `FSDP_CHANGES.md`

