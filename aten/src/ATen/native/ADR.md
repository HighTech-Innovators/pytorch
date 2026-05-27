# Architecture Decision Record: ATen Native Kernel Implementation

**Location:** `./src/aten/src/ATen/native/`  
**Last Updated:** 2026-05-27  
**Classification:** Performance Critical, Implementation Detail

---

## Executive Summary

ATen native kernels are the actual implementations of tensor operations. The `native/` directory contains hundreds of C++ files implementing operations like addition, matrix multiplication, convolutions, activation functions, and reshaping. Operations are organized by category, with separate subdirectories for device-specific implementations (CPU, CUDA, Metal, etc.).

---

## Architectural Role

Native kernels serve as the **actual operation implementation layer** for all tensor computations. Their responsibilities are:

1. **Operation Implementation** — Implement the mathematical logic for each operation
2. **Device Dispatch** — Provide device-specific implementations (CPU, CUDA, Metal)
3. **Performance Optimization** — Tune kernels for specific hardware (SIMD, CUDA cores, etc.)
4. **Library Integration** — Call optimized libraries (BLAS, CuBLAS, CuDNN) when beneficial
5. **Fallback Logic** — Provide CPU fallback when device-specific kernel unavailable
6. **Vectorization** — Use SIMD instructions (AVX, AVX2, SVE, etc.) on CPU

The quality and performance of user computations depends directly on the quality of native kernels. Slow kernels directly translate to slow training.

---

## Responsibilities

### What Native Kernels Do

- **Mathematical Computation** — Implement forward pass logic for all operations
- **Element-Wise Operations** — add, mul, sin, exp, relu, etc. (simple element-wise math)
- **Reduction Operations** — sum, mean, max, etc. (reduce along dimensions)
- **Matrix Operations** — matmul, transpose, reshape, etc. (matrix/array operations)
- **Convolutions** — 1D/2D/3D convolutions, dilated convolutions, grouped convolutions
- **Normalization** — batch norm, layer norm, etc.
- **Pooling** — max pool, average pool, adaptive pooling
- **Library Calls** — Call CuBLAS for matmul on CUDA, CuDNN for convolutions, etc.
- **CPU Fallback** — Provide pure C++ CPU implementation when no device-specific version
- **Optimization** — Use SIMD, memory layout optimization, kernel fusion where beneficial

### What Native Kernels Do NOT Do

- **Autograd** — Backward pass not handled here; each operation's backward is registered separately
- **Dispatch** — Operation selection/routing handled by ATen dispatch system
- **Memory Allocation** — Memory allocated by tensor creation; kernels assume tensors already exist
- **Type Checking** — Type compatibility checked before kernel is called
- **Device Context** — Device is pre-determined; kernels don't change device context

---

## Dependencies

### What Depends On Native Kernels

- **ATen Dispatcher** — Routes operations to correct native kernel
- **All User Code** — Every operation ultimately calls a native kernel
- **torch/** — Python operations call ATen which calls native kernels
- **Autograd** — Records which kernel executed for backward pass

### What Native Kernels Depend On

- **ATen Core** — Uses Tensor, Storage types from c10/core
- **BLAS Libraries** — CPU kernels call Cblas; CUDA kernels call CuBLAS
- **CuDNN** — CUDA convolution kernels call CuDNN
- **Device APIs** — CUDA kernels call CUDA runtime API
- **Third-Party Libraries** — May call Eigen, oneDNN, etc. for optimizations

**Dependency Direction**: Unidirectional upward. Everything depends on native kernels; they depend downward on c10/core and external libraries.

---

## Trade-Offs and Design Decisions

### Decision 1: Organize by Operation Category vs By Device

**What**: Organize at directory level by operation category (Convolution/, MatrixAlgebra/, Activation/), with device-specific subdirectories (cpu/, cuda/, metal/) underneath each category.

**Why**:
- Code Locality: All convolution logic (CPU, CUDA, Metal) in same directory tree
- Reduced Duplication: CPU fallback is in same file as CUDA implementation
- Single Registration Point: Dispatch registration for all variants in one file
- Development Workflow: Developers working on convolutions don't need to navigate multiple device folders

**Alternatives**:
- Organize by Device First — `cpu/`, `cuda/`, `metal/` with all operations inside (like MLPerf)
- Monolithic File Per Operation — all implementations in single file (wouldn't scale)

**Trade-offs**:
- **Pro**: Logical organization by operation; easier to understand what a component does
- **Con**: Directory structure varies based on how operations naturally cluster; not uniform

---

### Decision 2: Library Calls (CuBLAS, CuDNN) vs Pure Implementation

**What**: For common operations (matmul, convolution), call optimized libraries instead of implementing custom kernels.

**Why**:
- Performance: CuBLAS and CuDNN are highly optimized by NVIDIA; hard to beat
- Maintenance: Let vendors maintain library code; focus effort on logic we control
- Bug Fixes: Security patches and performance improvements come automatically with library updates

**Alternatives**:
- Implement All Operations from Scratch — would require massive engineering effort; hard to match library performance
- Use Framework Default — TensorFlow uses TensorFlow Lite for some ops; PyTorch chose vendor libraries

**Trade-offs**:
- **Pro**: Best performance; reduced maintenance burden
- **Con**: Library versions matter; dependency on external libraries; license compliance needed

---

### Decision 3: CPU SIMD Vectorization via Vectorized Dispatch

**What**: CPU operations use `cpu/vec/` infrastructure that dispatches to vectorized code paths (AVX, SSE, etc.) at runtime.

**Why**:
- Portability: Single code compiles; runtime dispatch selects fastest available vector width
- Performance: AVX-512 code is 8x faster than scalar for float32; can't ignore vectorization
- Binary Size: Compile once; don't ship multiple binaries for different CPUs

**Alternatives**:
- Scalar Only — simple but slow; would handicap performance
- Multiple Binaries — ship CPU-specific binaries for different vector widths; hard to distribute
- Compile-Time Dispatch — require build-time CPU specification (not portable)

**Trade-offs**:
- **Pro**: Good performance; portable; single binary
- **Con**: Runtime dispatch adds complexity; vectorization code harder to debug

---

### Decision 4: CPU Fallback Strategy

**What**: Every operation has a CPU-only implementation as fallback; device-specific kernels (CUDA, Metal) delegate to CPU if needed.

**Why**:
- Correctness: Even on GPU, all operations must work; CPU fallback ensures this
- Debugging: CPU version provides simpler path for debugging GPU kernels
- Portability: Code works on all platforms even if optimal kernel missing for a device

**Alternatives**:
- Device-Only: Require all operations on GPU; fail if kernel not available (not portable)
- Lazy Implementation: Implement only common operations; others error out (breaks code)

**Trade-offs**:
- **Pro**: Robust; good for debugging; portable
- **Con**: CPU fallback may be slow; duplicated logic across CPU and GPU versions

---

## Extension Boundaries

### Extending Native Kernels

**Supported Extensions:**
1. **New Operation** — Add new operation registration and CPU kernel in `native/*.cpp`
2. **Device-Specific Optimization** — Override CPU kernel with CUDA-optimized version in `native/cuda/`
3. **Library Integration** — Call new library if available (e.g., integrating oneDNN)
4. **Custom Kernel** — Users can't directly extend; must go through ATen operator registration

**NOT Supported:**
- Changing kernel signature after registration
- Adding new device backend at runtime (requires recompilation)

### Integration Points

- **ATen Dispatch** — Via `TORCH_LIBRARY` registration
- **Device Allocators** — Via c10/core allocator interface
- **Autograd** — Via saved tensors and backward function registration
- **Libraries** — CuBLAS, CuDNN, BLAS via C/C++ interfaces

---

## Runtime Implications

### Initialization Order

**At Library Load:**
1. `aten/src/ATen/native/` — Load CPU implementations
2. `aten/src/ATen/native/cpu/vec/` — Initialize CPU vector width detection (AVX, SSE, etc.)
3. Device backends (cuda/, metal/) — Register device-specific kernels
4. Library Initialization — Initialize CuBLAS, CuDNN if available

**At First Operation:**
- Dispatch lookup selects correct kernel
- Kernel executes with pre-allocated tensor memory

### Concurrency and Thread Safety

- **Stateless Kernels** — Most kernels are pure functions; no shared state
- **BLAS Libraries** — Thread-safe; can be called concurrently from multiple threads
- **GPU Kernels** — CUDA operations are queued asynchronously; GPU handles synchronization
- **Vector Dispatch** — CPU vector width is detected once at startup; cached

### Failure Modes

- **Unsupported Operation on Device** — Operation not implemented for device; falls back to CPU (slow)
- **Numerical Overflow** — Integer overflow in element-wise ops not detected; silently wraps
- **Library Missing** — CuBLAS not available; operation fails with error
- **Memory Alignment** — Misaligned memory can cause SIMD issues; checked at kernel entry

---

## Performance Implications

### Hot Paths and Allocations

1. **Kernel Lookup** — O(1) hash table on operation name + device
2. **Memory Access** — Sequential memory access is fastest; random access causes cache misses
3. **SIMD Efficiency** — Vectorized operations achieve 8x-16x speedup vs scalar
4. **Library Calls** — CuBLAS/CuDNN handle actual computation; PyTorch just calls them

### Known Bottlenecks

- **CPU→GPU Transfer** — PCIe bandwidth limited; data transfer slower than computation
- **Unvectorized Ops** — Operations without SIMD can be 10-100x slower
- **Small Tensors** — Overhead of kernel dispatch larger than actual work for tiny tensors
- **Library Overhead** — CuBLAS kernel launch overhead significant for tiny operations

### Mitigation Strategies

- **Operation Fusion** — Combine multiple operations into single kernel to reduce overhead
- **Batching** — Process multiple independent operations as single kernel call
- **Memory Layout Optimization** — Keep tensors in memory layout matching kernel expectations
- **Tuning Parameters** — Configure CuBLAS/CuDNN hints for workload (e.g., beta=0 optimization)

---

## Ownership Boundaries

### What Native Kernels Own

- Actual operation implementations (the code that does math)
- Performance characteristics (which library is called, vectorization used)
- Correctness of computation (numerical accuracy)
- Error handling (when operations fail)

### What Native Kernels Do NOT Own

- Operation registration (ATen dispatch system)
- Type system (c10/core types)
- Memory management (allocators)
- Gradient computation (autograd)

### State Shared with Other Layers

- **Tensor Data** — Native kernels read/write tensor memory allocated elsewhere
- **Operation Semantics** — ATen defines what operation does; native kernels implement it
- **Performance Model** — Kernels report complexity; optimizer uses for memory/compute planning

---

## Testing and Validation

### Critical Tests

- **Numerical Correctness** — Verify computation matches expected mathematical result
- **Device Transfer** — Verify operations work on all supported devices
- **Edge Cases** — Zero-sized tensors, single-element tensors, large tensors
- **Backward Pass** — Verify autograd computes correct gradients
- **Vectorization** — CPU kernels produce correct results with AVX, SSE, etc.
- **Performance** — Benchmark kernels; alert if performance degrades

### Known Gaps

- Limited testing of numeric precision (float16, bfloat16)
- Limited testing of complex dtypes (complex64, complex128)
- No explicit testing of concurrent kernel calls
- Limited testing of library integration edge cases (CuBLAS version specific behavior)

---

## Related Systems

- **ATen Dispatch** — Routes operations to native kernels
- **c10/core/** — Provides types, allocators, device abstraction
- **Autograd** — Records which kernel executed; calls backward separately
- **Libraries** — CuBLAS, CuDNN, BLAS, Eigen, etc.

---

## References

- `aten/src/ATen/native/` — Main native kernel implementations
- `aten/src/ATen/native/cpu/` — CPU-specific optimizations
- `aten/src/ATen/native/cpu/vec/` — CPU SIMD vectorization
- `aten/src/ATen/native/cuda/` — CUDA kernel implementations (many delegate to CuBLAS/CuDNN)
- `aten/src/ATen/native/metal/` — Metal (Apple GPU) implementations
- Book Chapter 7 (Performance) — Discusses optimization strategies for kernels
- Book Chapter 4 (Tensor Core) — Explains memory layout and access patterns for kernels
