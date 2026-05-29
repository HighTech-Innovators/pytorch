# Architecture Decision Record: aten/src/ATen/native/cuda

## Architectural Role

`aten/src/ATen/native/cuda` provides CUDA kernel implementations of PyTorch operators, optimized for execution on NVIDIA GPUs. It complements `aten/src/ATen/native/cpu` for the GPU backend, handling kernel launches, GPU memory management coordination, and GPU-specific optimizations. This module is Runtime Critical for GPU-accelerated training and inference.

## Responsibilities

This module is responsible for:
- Implementing CUDA kernel code for operators (element-wise, reduction, matrix multiplication, attention, etc.)
- Managing CUDA kernel launches via the cuBLAS, cuDNN, and custom kernels
- Coordinating with c10/cuda allocator for GPU memory allocation and stream-aware synchronization
- Registering CUDA kernels with the dispatcher (via `TORCH_LIBRARY_IMPL(aten, CUDA, ...)`)
- Handling GPU stream selection and synchronization for multi-stream training
- Implementing mixed-precision kernels (float32, float16, bfloat16)
- Supporting sparse tensor operations and specialized GPU algorithms

The module does **not** implement CPU kernels, dispatch routing, or device memory allocation (that's c10/cuda's responsibility).

## Dependencies

**Inbound** (what depends on aten/src/ATen/native/cuda):
- Dispatcher: when a CUDA tensor operation is called, the dispatcher invokes CUDA kernels
- Python bindings for GPU operations
- `torch/cuda` for CUDA device management

**Outbound** (what aten/src/ATen/native/cuda depends on):
- `c10/cuda` for GPU memory allocation, stream management, and CUDA error handling
- NVIDIA libraries: cuBLAS, cuDNN, cuSPARSE, NCCL
- CUDA Toolkit (nvcc, runtime APIs)
- `aten/src/ATen/TensorIterator.h` for shape iteration (adapted for GPU)

## Trade-offs

**Custom CUDA kernels vs. library calls (cuBLAS/cuDNN)**: Complex operations like matrix multiplication and convolution use optimized library calls (cuBLAS, cuDNN) rather than custom kernels. This trades some flexibility for production-grade performance. Simple operations (element-wise) use custom kernels for lower latency.

**Single GPU stream per operation vs. custom stream management**: By default, operations run on a device's default stream. Advanced users can specify streams (e.g., for async data loading). This design is simple but forces synchronization at stream boundaries; custom stream management adds complexity.

**Kernel fusion via kernel templates vs. explicit fusion kernels**: Some operations can be fused (e.g., `add` + `relu`), but fusion is limited to elementwise operations. More complex fusions (e.g., `matmul` + custom activation) require custom kernels.

**Mixed-precision kernels**: Operations can run in float16 or bfloat16 to save memory and improve speed. The trade-off is numerical precision; bfloat16 has less precision than float16 but is easier to use in mixed-precision training.

## Extension Boundaries

- **Custom CUDA kernels**: Users can implement new operators using CUDA and register them with `TORCH_LIBRARY_IMPL(custom_namespace, CUDA, ...)`.
- **Library integration**: New GPU libraries can be integrated by linking their `.so` files and calling their APIs from kernel implementations.
- **Custom kernels for new hardware** (e.g., AMD MI250, Intel Arc): New GPU vendors can provide GPU-specific kernel implementations.

## Runtime Implications

**Kernel selection**: When a CUDA tensor operation is called, the dispatcher invokes the appropriate CUDA kernel from this module.

**Kernel launch**: CUDA kernels are launched asynchronously on the GPU. PyTorch typically does not synchronize after each kernel; synchronization occurs at Python-level barriers (e.g., `.item()`, `.cpu()`).

**Stream management**: Kernels run on the current CUDA stream (default stream per device). Custom stream selection is possible but requires explicit API usage.

**GPU memory coordination**: Before a CUDA kernel runs, its input tensors must be allocated on the same GPU device. The dispatcher enforces this; mismatched device tensors raise an error.

## Performance Implications

**GPU utilization**: CUDA kernels should fully utilize GPU resources (all SMs, memory bandwidth). Under-utilized kernels leave GPU idle, wasting expensive GPU time.

**Kernel launch overhead**: CUDA kernel launches add 1-5 microseconds overhead. For small operations, this overhead can exceed compute time. Kernel fusion (combining operations) reduces launch overhead.

**Memory bandwidth**: Many GPU operations are memory-bandwidth-limited (e.g., element-wise operations). Kernel performance depends on memory access patterns (coalescing, cache hits).

**Library overhead**: cuBLAS and cuDNN calls add overhead (~10-50 microseconds) compared to direct kernel invocation, but the superior algorithms typically offset this cost.

**GPU synchronization**: Implicit synchronization (e.g., moving tensors to CPU) can stall the GPU pipeline. Keeping data on GPU for as long as possible maximizes throughput.

## Ownership Boundaries

- **CUDA kernels own**: GPU computation (kernel code, registers, shared memory)
- **c10/cuda allocator owns**: GPU memory allocation and pooling
- **CUDA stream owns**: kernel execution order and implicit synchronization
- **cuBLAS/cuDNN own**: highly optimized implementations of complex operations

## Verification Points

- `aten/src/ATen/native/cuda/` — Directory of CUDA kernel implementations
- `c10/cuda/CUDACachingAllocator.h` — GPU memory allocation coordination
- `aten/src/ATen/cuda/` — CUDA-specific tensor utilities and error handling
- `torch/cuda/` — Python interface to CUDA operations
