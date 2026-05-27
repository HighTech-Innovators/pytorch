# Architecture Decision Record: torch.amp (Automatic Mixed Precision)

## Architectural Role

`torch.amp` is **PyTorch's mixed precision training infrastructure**, automatically selecting lower precision dtypes (float16, bfloat16) to improve performance while maintaining accuracy. It enables:

1. **Automatic dtype casting**: Operations cast to lower precision transparently
2. **Loss scaling**: Dynamic adjustment of loss scale to prevent underflow
3. **Backend selection**: Using device-specific optimizations (NVIDIA APEX, PyTorch native)
4. **Gradient computation**: Computing gradients at mixed precision
5. **Inference optimization**: Faster inference with reduced memory

Key insight: `torch.amp` is **automatic precision reduction**. It intercepts operations to decide which can run at lower precision (matmul, conv) and which must stay at higher precision (normalization, reductions). This trades precision for speed and memory.

## Responsibilities

### What This Subsystem Owns

1. **Autocast Context** (`torch.amp.autocast`)
   - Context manager for automatic dtype casting
   - Operation dispatch to appropriate dtype
   - Device-specific casting decisions

2. **Loss Scaling** (`torch.amp.GradScaler`)
   - Scaling loss to prevent gradient underflow
   - Tracking overflow in gradients
   - Dynamic adjustment of scale factor

3. **Operation Casting Rules** (`torch/amp/autocast_mode.py`)
   - Defining which operations can use lower precision
   - Per-operation dtype selection
   - Fallback to higher precision when needed

4. **Backend Integration**
   - CUDA-specific optimizations (TF32)
   - CPU-specific configurations
   - Other device support

### What This Subsystem Does NOT Own

- **Tensor implementations**: ATen owns storage and kernels
- **Automatic differentiation**: torch.autograd owns gradient computation
- **Memory management**: c10 allocators handle memory
- **Device management**: torch.device and backends own this
- **Operation execution**: Backend kernels execute operations

## Dependencies

### Upstream Dependencies (What Uses This)

- **User training code**: Context manager around training loops
- **Optimization frameworks**: Integration with training pipelines
- **Research code**: Performance-critical deep learning

### Downstream Dependencies (What This Uses)

- **torch.autograd**: For gradient computation at mixed precision
- **ATen operations**: Executing operations at selected dtypes
- **torch.cuda**: For device-specific features (TF32)

### Dependency Direction

```
User Training Code
    ↓
torch.amp (mixed precision)
    ├─→ Autocast (dtype selection)
    ├─→ GradScaler (loss scaling)
    └─→ ATen operations at selected dtypes
        ↓
    GPU/CPU execution with appropriate precision
```

## Trade-offs and Design Decisions

### Automatic vs. Manual Precision Selection

**Decision**: Automatically select precision based on operation type.

**Trade-off**:
- ✅ **Advantage**: Easy to use; no manual casting required
- ✅ **Advantage**: Consistent decisions across code
- ❌ **Disadvantage**: Users lose control over specific operations
- ❌ **Disadvantage**: Automatic decisions may not be optimal for all models

**Evidence**: `autocast` automatically casts operations; users don't manually select dtypes.

### Loss Scaling for Gradient Underflow

**Decision**: Use loss scaling to prevent gradient underflow at lower precision.

**Trade-off**:
- ✅ **Advantage**: Enables training at lower precision without numerical issues
- ✅ **Advantage**: Dynamic scaling adapts to gradient magnitudes
- ❌ **Disadvantage**: Additional hyperparameter to tune
- ❌ **Disadvantage**: Overflow detection adds overhead

**Evidence**: `GradScaler` scales losses during backward pass.

### Device-Specific Optimizations (TF32)

**Decision**: Use device-specific precision formats (TF32 on NVIDIA Ampere) automatically.

**Trade-off**:
- ✅ **Advantage**: Leverages hardware capabilities
- ✅ **Advantage**: Significant speedups on compatible devices
- ❌ **Disadvantage**: Behavior differs across devices
- ❌ **Disadvantage**: Users may not understand why performance differs

**Evidence**: TF32 enabled by default on NVIDIA GPUs.

## Runtime Implications

### Lifecycle and Initialization

1. **Context entry**: `with torch.amp.autocast():`
2. **Operation dispatch**: Each operation checked for lower precision eligibility
3. **Dtype casting**: Operations cast to float16 or bfloat16 as needed
4. **Execution**: Operations run at selected precision
5. **Result**: Mixed precision output
6. **Loss scaling**: GradScaler scales loss for backward
7. **Backward**: Gradients computed at mixed precision
8. **Unscaling**: GradScaler unscales gradients before optimizer update

### Dtype Selection Logic

- **Matmul/Conv**: Usually run at lower precision (float16/bfloat16)
- **Accumulation/Normalization**: Run at higher precision (float32)
- **Fallback**: If operation not supported at lower precision, run at float32

## Performance Implications

### Known Hotspots

1. **Autocast decision overhead**: Checking operation type and selecting dtype
2. **Dtype conversion**: Converting inputs to/from lower precision
3. **Loss scaling overhead**: Computing scaling factor and unscaling
4. **Reduced precision arithmetic**: May have lower throughput on some operations

### Speedup Opportunities

- **Tensor cores**: Dedicated hardware for lower precision matrix multiplication
- **Memory bandwidth**: Lower precision reduces memory bandwidth requirements
- **Reduced precision bias**: Some operations complete faster at lower precision

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/amp/autocast_mode.py` | Autocast context implementation |
| `torch/amp/grad_scaler.py` | Loss scaling implementation |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Examined `torch/amp/` implementation
2. **Documentation**: PyTorch AMP guide
3. **Code Flow**: Understanding autocast and grad scaling

Last Verified: 2026-05-27
