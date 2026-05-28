# ADR: ATen Tensor Operations Library

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

ATen (A Tensor Library) is PyTorch's primary tensor operations library and implements the dynamic dispatch system that routes all operator calls to backend-specific kernel implementations. It is the largest and most performance-critical subsystem in PyTorch (1,968 source files, 255 entrypoints).

ATen is classified as **Runtime Critical** and **State Owner**: it owns the operator registry, dispatcher implementation, and all kernel implementations for tensor operations. Nearly all computation in PyTorch flows through ATen's dispatch system.

## Responsibilities

### What ATen Owns
- **Operator registry and dispatcher**: Central registry mapping operator names to kernel implementations across all backends (CPU, CUDA, Sparse, Quantized, etc.)
- **Kernel implementations**: ~2,500 tensor operation kernels for different device types, tensor layouts, and data types
- **Native functions schema**: Operation type signatures and metadata (from native_functions.yaml)
- **Dispatch mechanism**: DispatchKeySet extraction, dispatch table indexing, fallback chains, boxing/unboxing
- **Memory allocation and caching**: Allocator implementations (CPUAllocator, CUDACachingAllocator, custom allocators)
- **Tensor factory functions**: randn(), zeros(), from_blob(), etc.
- **Core tensor math**: add, mul, matmul, convolution, and 2,500+ other operations

### What ATen Does Not Own
- **Python bindings**: Pybind11 wrappers are owned by torch/csrc
- **Automatic differentiation**: Autograd engine is owned by torch/csrc/autograd (though ATen dispatcher provides hooks for autograd kernels)
- **Neural network abstractions**: torch.nn modules are owned by torch/nn
- **High-level APIs**: torch.* convenience functions are owned by torch/
- **Core abstractions**: TensorImpl, Storage, Device, DispatchKey are owned by c10

## Dependencies

### Internal Dependencies (ATen → other modules)
- **c10**: Every file includes c10 headers (TensorImpl, Device, DispatchKey, intrusive_ptr)
- **CUDA toolkit** (optional): cuda/ subdirectory includes CUDA headers when CUDA support is enabled
- **oneDNN** (optional): Used for optimized CPU kernels (linear algebra, convolution)
- **XNNPACK** (optional): Mobile-optimized kernels

### External Dependencies (other modules → ATen)
- **torch/csrc depends on ATen**: Python bindings call ATen operators
- **torch/autograd depends on ATen**: Autograd engine registers gradient kernels and reads dispatch metadata
- **torch/nn depends on ATen** (transitively): Modules invoke ATen operators
- **torch/jit depends on ATen**: TorchScript graph executor invokes ATen kernels

**High fan-in**: ATen is imported by ~1,500+ files across the repository. Changes to dispatcher or core kernel signatures trigger cascading recompiles.

## Trade-offs and Design Decisions

### 1. Dynamic Dispatch vs Virtual Methods
**Decision**: Use dispatch table lookup instead of virtual method dispatch on Tensor  
**Rationale**:
- **Open-world extensibility**: New backends can register kernels without modifying ATen source
- **Per-backend customization**: Different backends can optimize independently (CPU vectorization vs CUDA memory coalescing)
- **Composability**: Multiple dispatch keys active simultaneously (Autograd + CPU + Profiler + Meta)
- **ABI stability**: Dispatch tables can be extended without breaking binary compatibility

**Trade-off**: More complex dispatcher code, but gains far outweigh implementation cost

**Evidence**: `aten/src/ATen/core/dispatch/Dispatcher.h:71` defines Dispatcher singleton. `aten/src/ATen/core/DispatchKey.h` lists 30+ dispatch keys. `aten/src/ATen/core/dispatch/OperatorEntry.h:65` shows dispatch table structure.

### 2. Separate Schema (TORCH_LIBRARY) and Implementation (TORCH_LIBRARY_IMPL) Registration
**Decision**: Operators are defined once in native_functions.yaml, then implemented across multiple backends  
**Rationale**:
- **Decoupling**: Operator interface (schema) is independent of backend implementations
- **Discoverability**: All operators listed in single file (native_functions.yaml); implementations scattered by backend
- **Validation**: Dispatcher validates kernel signatures match schema at registration time
- **Lazy loading**: Backends can register on-demand; schemas always present

**Trade-off**: Two-phase registration adds complexity but enables modular backend development

**Evidence**: `aten/src/ATen/templates/RegisterSchema.cpp` registers schemas. `aten/src/ATen/native/native_functions.yaml` (16,222 lines) defines all operators. Backend registration occurs via `TORCH_LIBRARY_IMPL(aten, CPU, m) { ... }` patterns in ~100 source files.

### 3. Boxing and Unboxing Kernels
**Decision**: Support both typed (unboxed) and type-erased (boxed) calling conventions  
**Rationale**:
- **Unboxed kernels** (typed C++ function pointers): Zero boxing overhead, optimal performance
- **Boxed kernels** (stack-based): Support dynamic dispatch from Python/TorchScript without compile-time type information
- **Transparent conversion**: KernelFunction converts between modes automatically

**Trade-off**: Adds indirection layer, but necessary for cross-language boundaries

**Evidence**: `aten/src/ATen/core/boxing/KernelFunction.h:84` implements conversion. `Dispatcher::callBoxed()` (line 851) uses boxed convention. `Dispatcher::call()` templates use unboxed convention.

### 4. Fallback Chains for Missing Kernels
**Decision**: If a kernel is missing for a dispatch key, try fallbacks (CompositeImplicitAutograd, backend fallbacks, etc.)  
**Rationale**:
- Reduces redundant kernel implementations (composite operators defined in terms of others)
- Enables sparse tensor operations (many kernels missing for sparse; fallback to dense)
- Supports new backends before all kernels are implemented

**Trade-off**: Fallback lookup adds overhead for missing kernels; optimized via caching

**Evidence**: `aten/src/ATen/core/DispatchKey.h` lists fallback keys. `aten/src/ATen/core/dispatch/OperatorEntry.cpp:81` implements fallthrough behavior.

### 5. RecordFunction Instrumentation at Every Dispatch Point
**Decision**: Profiler guard (RecordFunction) wraps every kernel call, even when profiling disabled  
**Rationale**:
- Enables detailed profiling of all operations without recompilation
- Hooks for TorchScript execution, distributed tracing
- Minimal overhead when profiling disabled (~5-10ns)

**Trade-off**: Constant overhead at hot dispatch paths, but acceptable for framework flexibility

**Evidence**: `aten/src/ATen/core/dispatch/Dispatcher.h:650` applies RecordFunction guards. `aten/src/ATen/record_function.h` defines mechanism.

## Extension Boundaries

### Public Extension Points
1. **Custom backends**: Register `TORCH_LIBRARY_IMPL(aten, CustomKey, m) { ... }` to implement kernels for new device types (e.g., XLA, MPS, Vulkan)
2. **Custom operators**: Use `TORCH_LIBRARY(mylib, m) { ... }` to define new operators that integrate with dispatcher
3. **Fallback registration**: Register global fallbacks for new dispatch keys (e.g., quantization wrapper)
4. **Custom allocators**: Implement and register Allocator interface for custom memory management

### Extension Constraints
- **native_functions.yaml is authoritative**: Custom operator schemas should not conflict with existing definitions
- **Dispatch key space is finite**: Cannot create custom dispatch keys at runtime (enum is compile-time)
- **Operator names are global**: Registration fails if operator already defined elsewhere
- **Kernel signature must match schema**: Dispatcher validates at registration time

## Runtime Implications

### Initialization
- **Static initialization order**: Schemas registered first (via `TORCH_LIBRARY`), then implementations (via `TORCH_LIBRARY_IMPL`)
- **Dispatcher singleton created** on first access (thread-safe lazy initialization)
- **All kernels pre-registered** by the time Python imports `torch`

### Dispatch Sequence
1. **Argument preparation**: Python → pybind11 → C++ types
2. **DispatchKeySet extraction**: Read tensor device, dtype, mode flags (5-10ns)
3. **Dispatch table lookup**: Index into operator's kernel array (5ns)
4. **Kernel invocation**: Call function pointer (2-5ns)
5. **Fallback handling** (if needed): Try next priority key (adds 10-100ns if key missing)

**Total dispatch overhead**: ~10-20ns per call (negligible for ops >1μs, measurable for microbenchmarks)

### Thread Safety
- **Dispatcher is thread-safe**: Internal mutex protects registration operations
- **Concurrent kernel calls**: Multiple threads can simultaneously invoke kernels; no global locks in hot path
- **RecordFunction callbacks**: Profiler hooks must be thread-safe; per-thread queues prevent contention

### Performance Characteristics
- **Dispatch table overhead**: ~1.6KB per operator (200 dispatch keys × 8 bytes); ~4MB total for 2,500 operators
- **Kernel lookup**: O(1) array indexing; CPU cache-friendly for hot operators
- **Fallback chain**: Variable cost (1-20ns per key tried); cached in JIT for static graphs

## Performance Implications

### Memory Overhead
- **OperatorEntry**: ~8KB per operator (schema + dispatch table + metadata)
- **Kernel function pointers**: 8 bytes each; ~1.6KB per operator for 200 dispatch keys
- **Total registry size**: ~20MB for 2,500 operators (acceptable for flexibility gained)

### Performance-Critical Paths
1. **Operator dispatch** (every tensor operation)
   - Bottleneck: DispatchKeySet extraction and table lookup
   - Optimization: Inlined `Dispatcher::singleton()`, cached dispatch in JIT
2. **Python-to-C++ boundary** (Python API calls)
   - Bottleneck: Argument marshaling (boxing), GIL management
   - Optimization: pybind11 optimizations, avoid Python callbacks from C++
3. **Memory allocation** (tensor creation)
   - Bottleneck: Allocator mutex contention, cache fragmentation
   - Optimization: Pre-allocation pools (not yet implemented), batch allocation
4. **Kernel execution** (varies by operation)
   - Bottleneck: Cache locality (CPU), memory bandwidth (CUDA)
   - Optimization: Backend-specific kernel tuning, fusion via TorchScript

### Synchronization Costs
- **Dispatcher registration lock**: Held during kernel registration (milliseconds at startup, not in hot path)
- **Allocator lock** (CPUAllocator): Acquired per allocation; potential contention under high-frequency allocation
- **RecordFunction queue** (per-thread): Lock-free queue design minimizes profiling overhead

## Ownership Boundaries

### Operator Registry Ownership
- **Schema ownership**: c10 core library (function signatures)
- **Implementation ownership**: Backend-specific directories (aten/src/ATen/native/cpu/, aten/src/ATen/native/cuda/)
- **Dispatcher ownership**: aten/src/ATen/core/dispatch/
- **Registration management**: Torchgen (code generation from native_functions.yaml)

### Kernel Registration Lifetime
- Kernels are registered during static initialization (before main())
- Lifetime extends throughout program execution
- Deregistration is rare (custom backends unloading)

## Notes and Caveats

1. **native_functions.yaml is large and frequently modified**: Any operator signature change requires regenerating all bindings and registrations (can trigger full rebuilds of dependent code)
2. **Dispatch key priority is critical**: Autograd keys must have higher priority than backend keys; mode keys (profiling, no_grad) must have appropriate priority
3. **Fallback resolution is complex**: If an operator has no kernel for a dispatch key, fallback chain is tried (CompositeImplicitAutograd → CompositeExplicitAutograd → backend fallback → error)
4. **Boxing overhead is measurable**: For microbenchmarks (<1ns operations), boxing overhead dominates; use unboxed calling convention for performance-critical code
5. **RecordFunction hooks are global**: All profiling callbacks run in the hot path; expensive callbacks degrade performance significantly
6. **Memory allocator is a potential bottleneck**: High-frequency tensor creation (many small tensors) can contend on allocator lock
7. **Open-world dispatch comes with responsibility**: External backends must implement thousands of kernels or provide comprehensive fallbacks; incomplete implementations lead to runtime errors at first use
