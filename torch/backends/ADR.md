# Architecture Decision Record: torch.backends

## Architectural Role

The `torch.backends` subsystem provides **backend-specific configuration and capabilities**. It enables users to:

1. **Query backend availability**: Check if CUDA, MPS, or other accelerators are available
2. **Configure backend behavior**: Enable/disable optimizations, set precision, reproducibility modes
3. **Access backend metadata**: Query device properties and capabilities

Key insight: torch.backends is a **thin user-facing configuration layer** — most device logic lives in c10/ATen/torch.cuda; this module provides convenient access to it.

## Responsibilities

### What This Subsystem Owns

1. **CUDA Backend Configuration** (`backends/cuda/`)
   - Availability check: `torch.backends.cuda.is_available()`
   - Global enabled/disabled flag
   - Matmul precision settings (TF32, FP32, etc.)
   - Stream synchronization options

2. **cuDNN Configuration** (`backends/cudnn/`)
   - Benchmark mode: auto-tune convolution algorithms
   - Deterministic mode: guarantee reproducibility
   - Enabled flag
   - Version information

3. **MPS Configuration** (`backends/mps/`)
   - Apple Metal Performance Shaders availability
   - Configuration flags

4. **Other Backends** (`backends/xpu/`, etc.)
   - Intel XPU and other accelerator configs

### What This Subsystem Does NOT Own

- **Device Implementation**: c10 owns device abstraction
- **Kernel Execution**: ATen owns operation kernels
- **Memory Management**: c10 allocators handle device memory
- **Distributed Training**: torch.distributed handles multi-device coordination

## Dependencies

### Upstream Dependencies

- **User Configuration Code**: Setting backend options before training

### Downstream Dependencies

- **torch.cuda**: Detailed CUDA configuration
- **c10**: Device type enumeration and properties
- **ATen**: Backend-specific kernel selection

## Trade-offs and Design Decisions

### Global Configuration vs. Per-Device

**Decision**: Backend configuration is mostly global (affects all operations) rather than per-device.

**Trade-off**:
- ✅ **Advantage**: Simple; single configuration point
- ❌ **Disadvantage**: Can't have different settings for different devices
- ❌ **Disadvantage**: Less fine-grained control

**Evidence**: `torch.backends.cudnn.enabled` is a global flag, not per-GPU.

## Runtime Implications

### Initialization

1. **Import**: Backend flags initialized at module load
2. **Configuration**: User code calls `torch.backends.cuda.is_available()` etc.
3. **Effect**: Configuration affects subsequent operations

### Concurrency Behavior

**Thread Safety**: Configuration flags are global; concurrent modification unsafe.

## Key Implementation Files

| File | Purpose |
|---|---|
| `backends/cuda/__init__.py` | CUDA backend configuration and availability |
| `backends/cudnn/__init__.py` | cuDNN configuration |
| `backends/mps/__init__.py` | MPS backend configuration |

Last Verified: 2026-05-27
