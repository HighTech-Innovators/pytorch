# Architecture Decision Record: aten/src/ATen/native/cpu

## Architectural Role

`aten/src/ATen/native/cpu` provides CPU-specific implementations of PyTorch operators, optimized for multi-threaded execution on CPUs via vectorization (SIMD) and work-stealing parallelism. It is isolated from other platform-specific kernels to enable compilation with aggressive CPU optimization flags (e.g., `-mavx2`). This module is Runtime Critical for CPU training and inference, providing the computational backend for non-GPU workloads.

## Responsibilities

This module is responsible for:
- Implementing operator kernels optimized for multi-core CPU execution
- Providing `cpu_kernel` and `cpu_kernel_vec` abstractions for both auto-vectorized and explicit SIMD implementations
- Managing thread parallelization via TensorIterator's `for_each()` work-stealing scheduler (with GRAIN_SIZE = 32768 elements)
- Implementing vectorized operations using the `Vectorized<T>` abstraction (AVX2, AVX-512)
- Registering CPU kernels with the dispatcher (via `TORCH_LIBRARY_IMPL(aten, CPU, ...)`)
- Supporting both scalar and vectorized code paths, with fallback to scalar for incompatible data types

The module does **not** implement dispatch routing or define operator schemas. It also does not handle other hardware backends (CUDA, MPS, etc.).

## Dependencies

**Inbound** (what depends on aten/src/ATen/native/cpu):
- Dispatcher: when a CPU tensor operation is called, the dispatcher invokes CPU kernels
- Python bindings for CPU tensor operations
- `aten/src/ATen/native` base implementations (some CPU kernels are variants of generic kernels)

**Outbound** (what aten/src/ATen/native/cpu depends on):
- `aten/src/ATen/TensorIterator.h` for shape iteration and threading abstraction
- `aten/src/ATen/native/utils_cpu.h` for CPU-specific utilities
- BLAS libraries (OpenBLAS, MKL) for matrix operations
- Standard library (OpenMP or pthread for threading)

## Trade-offs

**Explicit SIMD vs. auto-vectorization**: The module provides both `cpu_kernel` (relying on compiler auto-vectorization) and `cpu_kernel_vec` (explicit SIMD via `Vectorized<T>`). Explicit SIMD is faster but harder to write and maintain. The default should be auto-vectorization; explicit SIMD is used only for performance-critical paths.

**TensorIterator parallelism with fixed GRAIN_SIZE**: The module uses TensorIterator's `for_each()` with GRAIN_SIZE = 32768 elements as the threshold for parallel execution. Smaller operations run single-threaded (avoiding synchronization overhead), larger ones use work-stealing parallelism. This heuristic is fixed but can be overridden per-call. The trade-off is simplicity vs. per-operation tuning.

**CPU-specific compilation flags**: This directory is compiled with `-mavx2` (or `-O3 -march=native`), while other directories are not. This isolation enables aggressive optimization for CPU but complicates the build system.

## Extension Boundaries

- **Custom CPU kernels**: Users can implement new operators by registering kernels with `TORCH_LIBRARY_IMPL(custom_namespace, CPU, ...)`, using TensorIterator and Vectorized abstractions.
- **Vectorization patterns**: New vectorized operations can be added following `cpu_kernel_vec` patterns.
- **Parallelism strategies**: Advanced users can customize `for_each()` with custom grain sizes or scheduling strategies.

## Runtime Implications

**Kernel selection**: When a CPU tensor operation is called, the dispatcher invokes the appropriate CPU kernel from this module.

**Parallelism**: Operations larger than GRAIN_SIZE (32768 elements) are executed in parallel across available CPU cores via work-stealing. Smaller operations run serially to avoid synchronization overhead.

**Vectorization**: CPU kernels are compiled with SIMD flags enabled, enabling automatic vectorization (8-16 elements per cycle for float32). Explicit `cpu_kernel_vec` paths provide further control.

**Cache locality**: TensorIterator chunks work to maximize L2/L3 cache utilization during multi-threaded execution.

## Performance Implications

**Multi-threading breakpoint**: The GRAIN_SIZE = 32768 threshold means:
- Float32 tensors with 32768 elements = 128 KB (approx. half of L2 cache)
- Operations ≤ 128 KB: run single-threaded (no sync overhead)
- Operations > 128 KB: run in parallel (amortize thread spawn over more work)

**Vectorization**: Vectorized operations can process 8-16 elements per cycle (AVX2/AVX-512). Auto-vectorization is usually effective; explicit SIMD adds 2-5% performance for critical paths.

**Memory bandwidth**: Many CPU operations are memory-bandwidth-limited rather than compute-limited. Kernel performance depends more on memory access patterns than algorithmic complexity.

**OpenMP/pthread overhead**: Thread creation and synchronization add 1-10 microseconds overhead. This cost is negligible for operations processing millions of elements but significant for small tensors.

## Ownership Boundaries

- **CPU kernels own**: algorithm implementation for CPU execution
- **TensorIterator owns**: shape iteration, parallelism scheduling
- **Vectorized<T> owns**: SIMD lane abstractions (but not actual CPU registers)
- **BLAS libraries own**: matrix multiplication and linear algebra implementations

## Verification Points

- `aten/src/ATen/native/cpu/Loops.h:1-55` — CPU kernel loop patterns and SIMD abstractions
- `aten/src/ATen/native/cpu/Loops.h:7-15` — GRAIN_SIZE and parallelism thresholds
- `aten/src/ATen/TensorIterator.h:77-78` — TensorIterator threading interface
- `aten/src/ATen/native/cpu/` — Directory of CPU kernel implementations
