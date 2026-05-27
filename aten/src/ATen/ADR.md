# Architecture Decision Record: ATen — A Tensor Library

**Location:** `./src/aten/src/ATen/`  
**Last Updated:** 2026-05-27  
**Classification:** Runtime Critical, State Owner, Core Operations

---

## Executive Summary

ATen (A Tensor Library) is the core computational engine of PyTorch. It implements all tensor operations (element-wise math, matrix multiplication, convolutions, etc.) and defines the public C++ API for tensor operations. Every operation a user performs on a tensor is ultimately routed through ATen's operator registry and dispatched to a device-specific kernel.

---

## Architectural Role

ATen serves as the **central operator hub and kernel dispatcher** for all tensor computations. Its responsibilities are:

1. **Operator Registration** — Define what operations are available (add, mul, conv2d, etc.)
2. **Type Dispatch** — Route operations to correct kernel based on device and dtype
3. **Kernel Implementation** — Provide kernels for CPU, CUDA, and other backends
4. **Memory Management** — Allocate and manage tensor storage
5. **Tensor View Operations** — Support reshaping, slicing, indexing without copying

Because every tensor operation flows through ATen, **all performance characteristics and correctness properties ultimately depend on ATen's architecture**.

---

## Responsibilities

### What ATen Does

- **Operator Registry** — Stores 1000+ operations registered via `TORCH_LIBRARY` macro and C++ declarations
- **Dispatch Table** — Routes operations to device-specific kernels based on DispatchKey
- **Native Kernel Implementation** — Implements CPU kernels directly; CUDA kernels call device APIs
- **Memory Allocation** — Creates tensors via device allocators (defined in c10/core)
- **View and Shape Operations** — Implements reshape, transpose, slice, squeeze, unsqueeze without data copy
- **Automatic Broadcasting** — Broadcasts operands to compatible shapes
- **Type Promotion** — Automatically promotes types (e.g., int + float → float)

### What ATen Does NOT Do

- **Gradient Tracking** — Autograd is separate (torch/autograd/); ATen kernels don't know about gradients
- **JIT Compilation** — TorchScript is in torch/csrc/jit/
- **Python Bindings** — pybind11 bindings are separate; ATen is pure C++
- **Dynamic Computation Graphs** — Graph construction is torch/ responsibility

---

## Dependencies

### What Depends On ATen

- **torch/** — Python API wraps ATen operations; users call torch.add, which routes to aten::add
- **torch/autograd/** — Autograd records which ATen operation executed; doesn't call ATen directly
- **torch/csrc/jit/** — TorchScript compiles to ATen operations
- **All Device Backends** — CUDA, Metal, etc. implement ATen operation kernels

### What ATen Depends On

- **c10/core/** — Uses Device, Allocator, DispatchKey, TypeMeta for type system
- **pthreadpool** (third-party) — For CPU thread parallelization
- **Platform ABIs** — BLAS (CBLAS, CuBLAS), threading (pthreads)

**Dependency Direction**: Unidirectional upward. torch/ and device backends depend on ATen; ATen depends downward on c10/core.

---

## Trade-Offs and Design Decisions

### Decision 1: Operator Registration via Macros vs Type-Driven Dispatch

**What**: Operators are registered via `TORCH_LIBRARY()` macro in header files, not auto-derived from template specializations.

**Why**:
- Explicit Control: Developers can choose which dtype/device combos to support
- Generated Fallbacks: Can provide automatic CPU→CUDA fallbacks
- Performance Profiling: Clear mapping of operation name to implementation

**Alternatives**:
- Template Specialization — C++ would auto-generate kernel variants for all dtype/device combos (explosion of binary size)
- Runtime Type Inspection — Would require reflection API; slower dispatch

**Trade-offs**:
- **Pro**: Small binary, explicit control, cacheable builds
- **Con**: Requires manual registration; missing registrations are silent errors until runtime

---

### Decision 2: Dispatch Table with DispatchKeySet vs Single Device Switch

**What**: Operations dispatch based on a `DispatchKeySet` (combination of device, dtype, quantization, sparsity, etc.), not just device type.

**Why**:
- Layered Operations: Sparse tensors, quantized tensors, and autograd-tracked tensors need special handling
- Composable: Layers can be combined (sparse + quantized, autograd + sparse, etc.)
- Extensibility: New operation types (functorch transforms) can add their own dispatch keys

**Alternatives**:
- Single Device Switch — only check device type; all special cases handled via type checks inside kernel
- Virtual Methods on TensorImpl — slower; less composable

**Trade-offs**:
- **Pro**: Flexible; supports many operation categories; enables functorch and other extensions
- **Con**: Complex dispatch logic; requires understanding multiple layers

---

### Decision 3: Kernel Implementation in Native vs Separate Backend Modules

**What**: Some operations (add, mul, basic shapes) are implemented directly in `aten/native/`; CUDA operations call into CuBLAS, CuDNN, etc.

**Why**:
- CPU Portability: Basic ops should work on all devices; implement once
- GPU Efficiency: CUDA operations delegate to optimized libraries
- Code Organization: Kernels organized by operation type, not device

**Alternatives**:
- All Operations in Device Folders — would duplicate common logic across CPU, CUDA, Metal, etc.
- All Operations as Thin Device Wrappers — inappropriate for ops that are inherently device-agnostic

**Trade-offs**:
- **Pro**: Reduces code duplication; single implementation for all devices where feasible
- **Con**: Some ops have device-specific optimizations requiring special cases

---

### Decision 4: Automatic Broadcasting and Type Promotion

**What**: Binary operations (add, mul, etc.) automatically broadcast operands to compatible shapes and promote types (int + float → float).

**Why**:
- NumPy Compatibility: Mirrors NumPy broadcasting semantics
- User Convenience: Avoids explicit reshape/cast in many scenarios
- Correctness: Type promotion prevents integer overflow bugs

**Alternatives**:
- Strict Type System — require exact type/shape match; force users to cast
- Lazy Broadcasting — defer broadcast decision to kernel; kernel checks shapes at runtime

**Trade-offs**:
- **Pro**: Convenient; matches NumPy expectations; prevents overflow bugs
- **Con**: Implicit cost (broadcasting may allocate temporary memory); not always intuitive

---

## Extension Boundaries

### Extending ATen

**Supported Extensions:**
1. **New Operation** — Register operation signature via `TORCH_LIBRARY`, provide implementations for each device
2. **New Device Backend** — Register dispatch keys for new device, provide kernels for all operations
3. **Device-Specific Optimization** — Override default CPU kernel with CUDA-specific kernel

**NOT Supported:**
- Modifying operation signatures after registration
- Changing operation semantics (e.g., adding new shape inference rules after tensor creation)

### Integration Points

- **c10/core/** — Via Device and DispatchKey for operation routing
- **torch/** — Via Python bindings; torch.add → aten::add
- **torch/autograd/** — Via operation hooks; autograd doesn't modify ATen operations
- **Device Backends** — Via dispatch key registration and kernel implementation

---

## Runtime Implications

### Initialization Order

**At Library Load:**
1. `aten/src/ATen/core/` — Initialize Tensor and type system
2. `aten/src/ATen/ops/` — Register operation stubs (generated by torchgen)
3. Device Backends — Register dispatch keys and kernel implementations

**At First Use:**
- Dispatch table is populated lazily on first operation call
- Allocators are selected based on device

### Concurrency and Thread Safety

- **Operation Registry**: Immutable after initialization; thread-safe reads
- **Dispatch Table**: Protected by locks during initialization; thread-safe reads after
- **Memory Allocation**: Allocators are thread-safe
- **Tensor Operations**: Most operations are thread-safe for independent tensors; in-place operations on same tensor from multiple threads are unsafe

### Failure Modes

- **Shape Mismatch**: Broadcasting fails if shapes are incompatible (exception)
- **Type Mismatch**: Type promotion fails for incompatible types (exception)
- **Missing Kernel**: Operation on unsupported device/dtype throws "not implemented" error
- **Out of Memory**: Allocator throws if insufficient memory

---

## Performance Implications

### Hot Paths and Allocations

1. **Dispatch Lookup** — Hash table lookup O(1) on operation name, then O(1) dispatch key set match
2. **Memory Allocation** — Device allocator call; may be cached depending on implementation
3. **Kernel Execution** — Depends on operation; matrix multiplication calls optimized BLAS

### Known Bottlenecks

- **Type Promotion** — Creates temporary tensors for broadcasting; can be expensive for large tensors
- **Memory Allocation** — First tensor of a given device type may trigger allocator initialization
- **Dispatch Indirection** — Virtual method calls for each operation (small fixed cost)

### Mitigation Strategies

- **In-Place Operations** — Avoid broadcasting by modifying tensor in place (`a += b`)
- **View Operations** — Use view/reshape/transpose to avoid copies
- **Kernel Fusion** — Combine multiple operations (e.g., `add + mul`) into single kernel

---

## Ownership Boundaries

### What ATen Owns

- Operation registry and dispatch table
- Operation semantics (shape inference, type promotion, broadcasting)
- CPU kernel implementations
- Memory allocation policy (which allocator is used for which device)

### What ATen Does NOT Own

- Device-specific allocator implementation (owned by device backends)
- Operation kernel implementation (owned by device-specific folders like `native/cuda/`)
- Autograd metadata (owned by torch/autograd/)
- Python bindings (owned by torch/)

### State Shared with Other Layers

- **Operation Registry** — Read by torch/, torch/autograd/, torch/jit/
- **Dispatch Table** — Populated by device backends; read by all operation callers
- **Tensor Shape/Type** — Shared with c10/core TensorImpl

---

## Testing and Validation

### Critical Tests

- **Dispatch Correctness**: Verify correct kernel is selected for each device/dtype
- **Broadcasting**: Verify shape compatibility and broadcasting rules
- **Type Promotion**: Verify integer + float → float, etc.
- **Memory**: Verify allocations are freed; no leaks
- **Device Transfers**: Verify tensors moved correctly between devices

### Known Gaps

- Limited testing of complex operation combinations
- No explicit performance regression tests for common operations
- Limited testing of error handling under memory pressure

---

## Related Systems

- **c10/core/** — Provides type system, device abstraction, and dispatch infrastructure
- **torch/** — Python API that wraps ATen operations
- **torch/autograd/** — Records which operations executed for gradient computation
- **Device Backends** — Implement kernels for specific operations on specific devices
- **torchgen/** — Generates operation stub code from declarations

---

## References

- `aten/src/ATen/core/Tensor.h` — Tensor wrapper class
- `aten/src/ATen/native/` — CPU kernel implementations
- `aten/src/ATen/native/cuda/` — CUDA kernel implementations
- Book Chapter 4 (Tensor Core) — Explains tensor structure and memory management
- Book Chapter 3 (Operator Dispatch) — Explains how operations are routed to kernels
