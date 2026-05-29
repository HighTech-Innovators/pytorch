# Architecture Decision Record: torch/cuda

## Architectural Role

`torch/cuda` provides the Python-level interface to NVIDIA CUDA, exposing GPU device management, memory statistics, and synchronization utilities. It is the primary API for GPU-related operations in user code and libraries.

## Responsibilities

- Implementing CUDA device management (torch.cuda.device, torch.cuda.is_available, torch.cuda.get_device_name)
- Exposing memory statistics (torch.cuda.memory_allocated, torch.cuda.memory_reserved, torch.cuda.memory_stats)
- Providing synchronization utilities (torch.cuda.synchronize, torch.cuda.stream)
- Managing CUDA context and device selection

## Dependencies

**Inbound**: User code, libraries
**Outbound**: `c10/cuda`, `aten/src/ATen/cuda`

## Trade-offs

**Lazy initialization**: CUDA is lazily initialized on first GPU operation, reducing startup time but delaying errors.

## Runtime Implications

**Device selection**: Current device is thread-local; switching devices changes context for subsequent operations.

**Memory tracking**: CUDA maintains per-device memory statistics accessible via `torch.cuda.memory_stats()`.

## Performance Implications

**API overhead**: Negligible; mostly thin wrappers around C++ functions.

## Ownership Boundaries

- **torch.cuda owns**: Python API
- **c10/cuda owns**: C++ implementation

## Verification Points

- `torch/cuda/__init__.py` — Public interface
- `torch/cuda/memory.py` — Memory management
