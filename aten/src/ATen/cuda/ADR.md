# Architecture Decision Record: aten/src/ATen/cuda

## Architectural Role

`aten/src/ATen/cuda` provides CUDA-specific utilities and abstractions that bridge PyTorch's tensor system with NVIDIA CUDA infrastructure. It supplies stream guards, event management, error translation, and device-specific properties required by GPU operations. This module is essential for GPU support but is smaller and more specialized than the native kernel implementations.

## Responsibilities

This module is responsible for:
- Implementing `CUDAStreamGuard` for managing CUDA stream context
- Providing CUDA device properties and capabilities queries
- Translating CUDA errors to PyTorch exceptions via `check_cuda_error`
- Managing CUDA events for stream synchronization and timing
- Providing device-specific initialization and cleanup
- Supporting device capabilities caching (compute capability, memory bandwidth, etc.)

The module does **not** implement tensor kernels (that's aten/src/ATen/native/cuda), memory allocation (that's c10/cuda), or dispatcher logic.

## Dependencies

**Inbound** (what depends on aten/src/ATen/cuda):
- `aten/src/ATen/native/cuda` for stream management and error handling in kernels
- `torch/cuda` (Python layer) for CUDA device properties
- High-level ATen operations that need CUDA utilities

**Outbound** (what aten/src/ATen/cuda depends on):
- `c10/cuda` for CUDACachingAllocator and basic CUDA wrappers
- NVIDIA CUDA runtime (cudaStreamCreate, cudaEventCreate, etc.)

## Trade-offs

**Stream guard RAII pattern**: CUDAStreamGuard uses RAII (acquire-on-construct, release-on-destruct) to manage stream context. This is exception-safe but requires careful scoping. The alternative (explicit save/restore) is more flexible but error-prone.

## Extension Boundaries

- **Device property extensions**: New CUDA device properties can be added to the capabilities cache.
- **Custom stream management**: Advanced users can create custom CUDA streams and pass them to operations.

## Runtime Implications

**Stream selection**: Operations run on the current CUDA stream, selected via CUDAStreamGuard. Default stream is used if no guard is active.

**Error handling**: CUDA errors are checked after kernel launches and translated to Python exceptions.

## Performance Implications

**Stream guard overhead**: Setting the current stream adds 1-2 cycles overhead but is typically amortized over kernel execution.

**Device property caching**: Properties are cached to avoid repeated CUDA driver calls.

## Ownership Boundaries

- **CUDAStreamGuard owns**: current stream context (during guard lifetime)
- **CUDA device owns**: actual GPU resources and memory

## Verification Points

- `aten/src/ATen/cuda/CUDAStreamGuard.h` — Stream context management
- `aten/src/ATen/cuda/CUDAException.h` — Error translation
