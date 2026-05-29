# Architecture Decision Record: aten/src/ATen/native

## Architectural Role

`aten/src/ATen/native` houses the bulk of PyTorch's operator kernel implementations—the computational logic that actually executes tensor operations. It provides CPU, CUDA, and device-agnostic kernels for hundreds of operators (add, matmul, convolution, attention, etc.), organized by dispatch key and compute pattern. This module is Runtime Critical; the performance and correctness of training and inference depend directly on the quality of kernels here.

## Responsibilities

This module is responsible for:
- Implementing operator kernels for all dispatch keys (CPU, CUDA, MPS, XLA fallbacks, etc.)
- Providing `TensorIterator` infrastructure that abstracts shape iteration and broadcasting, reducing kernel boilerplate
- Implementing structured kernels (those with explicit output allocation semantics)
- Providing kernel fusion utilities and pattern libraries (e.g., vectorized loops for CPU, CUDA kernel templates)
- Managing kernel registration via `TORCH_LIBRARY_IMPL(aten, CPU/CUDA/...)` macros
- Supporting operator variants (inplace, out-of-place, view-creating)
- Providing device-agnostic implementations (fallbacks that work on all backends)

The module does **not** manage dispatch routing (that's aten/src/ATen/core/dispatch) or define operator schemas (that's aten/src/ATen/core).

## Dependencies

**Inbound** (what depends on aten/src/ATen/native):
- `aten/src/ATen/core` for schema registration and dispatcher entry
- `torch/csrc/autograd` for autograd-aware kernel wrapping
- Python bindings for operator exposure

**Outbound** (what aten/src/ATen/native depends on):
- `c10/core` for TensorImpl, Device, DispatchKeySet
- `c10/cuda` for CUDA kernel launch infrastructure (for CUDA kernels)
- Device-specific libraries (BLAS, cuBLAS, NCCL, etc.)
- `aten/src/ATen/TensorIterator.h` for shape iteration abstractions

## Trade-offs

**TensorIterator abstraction for shape iteration**: Most kernels use TensorIterator to abstract over shape broadcasting, strides, and layout. This eliminates redundant iteration logic across hundreds of operators but adds small overhead (checking iteration parameters on each loop iteration). The alternative (hand-rolled iteration in each kernel) would be faster but more error-prone and maintainable.

**Device-agnostic implementations via fallbacks**: Some operators have single implementations registered to a fallback dispatch key (e.g., `Meta` for shape inference). These work on all backends but may be suboptimal for specialized ones. The trade-off is simplicity vs. device-specific optimization.

**Structured vs. unstructured kernels**: Structured kernels (those with explicit output allocation, derivatives, etc.) follow a template that enables automatic code generation and composition. Unstructured kernels are more flexible but require manual registration for each variant.

**Kernel fusion via TensorIterator**: When multiple operations can be fused into a single kernel, TensorIterator's abstraction enables elementwise fusion (e.g., `add` + `relu` fused into one kernel) but may not capture all possible fusion opportunities.

## Extension Boundaries

- **Custom operator kernels**: Users can implement new operators by registering kernels with `TORCH_LIBRARY_IMPL(custom_namespace, CPU/CUDA, ...)`. PyTorch's kernel infrastructure (TensorIterator, type conversion) is available to them.
- **Device-specific optimization**: New backends can register device-specific kernels for existing operators, overriding generic implementations.
- **Kernel patterns and templates**: Libraries can define new kernel patterns (vectorization strategies, memory layouts) that follow TensorIterator abstractions.

## Runtime Implications

**Kernel selection**: When an operator is called, the dispatcher selects the appropriate kernel from aten/src/ATen/native based on the tensor's dispatch key. CPU tensors invoke CPU kernels, CUDA tensors invoke CUDA kernels.

**Shape iteration**: TensorIterator setup happens at kernel entry time and involves examining tensor shapes, strides, and device. This setup cost is amortized over the kernel's loop iterations.

**Memory layout**: Kernels work with tensors in arbitrary memory layouts (contiguous, strided, etc.). TensorIterator handles this abstraction; kernels typically assume C-contiguous layout for simplicity.

**Backward compatibility**: Kernel implementations can vary across PyTorch versions (e.g., faster algorithms, precision improvements), but the operator API (name, signature) remains stable.

## Performance Implications

**Vectorization (CPU)**: CPU kernels in `aten/src/ATen/native/cpu` are compiled with AVX2 (or higher) flags, enabling SIMD vectorization for compatible operations.

**Kernel launch overhead (CUDA)**: Each CUDA kernel launch adds 1-5 microseconds of overhead. For small operations, this overhead can exceed computation time. Kernel fusion (combining multiple ops into one kernel) reduces launch overhead but is complex to implement.

**Shape iteration overhead**: TensorIterator's abstraction adds 1-2 cycles per loop iteration compared to hand-rolled iteration, but this is negligible for operations that compute significant work per element.

**Memory bandwidth**: Many operators are memory-bandwidth-limited rather than compute-limited. Kernel performance is often determined by memory access patterns more than algorithmic complexity.

## Ownership Boundaries

- **Native kernels own**: algorithm implementation (what computation to perform)
- **TensorIterator owns**: shape iteration, memory layout abstraction
- **TensorImpl owns**: the input/output tensor metadata (shape, dtype, device)
- **Dispatcher owns**: kernel routing and selection

## Verification Points

- `aten/src/ATen/native/*.cpp` — Individual operator implementations
- `aten/src/ATen/TensorIterator.h:77-78` — TensorIterator shape iteration interface
- `aten/src/ATen/native/cpu/Loops.h:1-55` — CPU kernel loop patterns
- `aten/src/ATen/TensorIterator.h:446-450` — TensorIterator operation chaining
