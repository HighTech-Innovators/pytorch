# ADR: c10 Core Library

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The c10 library ("core 10 functions") provides the foundational abstractions upon which all of PyTorch is built. It owns the runtime representation of tensors (TensorImpl), memory management abstractions (Storage, Allocator, Device), and the dispatch routing system (DispatchKey) that determines which kernel implementations execute operations.

c10 is classified as a **Lifecycle Root** and **State Owner**: it initializes before all other subsystems and defines the core data structures that persist throughout program execution.

## Responsibilities

### What c10 Owns
- **TensorImpl**: The runtime representation of every tensor, including storage references, shape metadata, device placement, dispatch keys, and autograd tracking state
- **Storage and memory management**: Buffer abstraction (Storage), allocator interface (Allocator), and device-specific allocation strategies (CPUAllocator, CUDACachingAllocator)
- **Device abstraction**: Representation of execution targets (CPU, CUDA, Meta) with type and index metadata
- **DispatchKey routing**: Enum and bitset machinery (DispatchKey, DispatchKeySet) that operators use to select backend-specific kernels
- **Reference counting**: Intrusive pointer mechanism (intrusive_ptr, intrusive_ptr_target) for efficient shared ownership
- **Basic utilities**: Array references (ArrayRef), optional wrappers, exception types, logging macros

### What c10 Does Not Own
- **Operator implementations**: Kernels are owned by ATen (aten/src/ATen/native/*)
- **Python bindings**: Pybind11 interface is owned by torch/csrc
- **Automatic differentiation**: Autograd engine is owned by torch/csrc/autograd (though TensorImpl carries autograd metadata)
- **Neural network abstractions**: torch.nn modules are owned by torch/nn
- **High-level APIs**: torch.* namespace and convenience functions are owned by torch/

## Dependencies

### Internal Dependencies (c10 → other modules)
- **None in the standard build** — c10 is intentionally dependency-free at the C++ level for maximum portability
- Optional: CUDA headers (cuda/CUDAFunctions.cpp) when CUDA support is enabled
- Optional: libfmt for formatting utilities

### External Dependencies (other modules → c10)
- **ATen depends on c10**: Every operator in aten/ includes c10 headers for TensorImpl, Device, DispatchKey
- **torch/csrc depends on c10**: Python bindings use TensorImpl, Storage, Device
- **torch/autograd depends on c10**: Autograd engine reads/writes autograd_meta_ on TensorImpl
- **torch/nn depends on c10** (transitively through aten): Modules work with Tensors built on TensorImpl

**High fan-in**: c10 is imported by nearly every compilable source file in PyTorch (conservative estimate: 500+ files). Changes to c10 public headers trigger full rebuilds of dependent subsystems.

## Trade-offs and Design Decisions

### 1. Intrusive Pointers vs std::shared_ptr
**Decision**: Use intrusive reference counting (c10::intrusive_ptr) instead of std::shared_ptr  
**Rationale**:
- Reduces memory overhead: Reference count lives *inside* TensorImpl (single allocation), not in a separate control block
- Enables atomic operations on reference counts without extra indirection
- Matches the cost model of C++ intrusive_ptr_target

**Trade-off**: Targets must inherit from intrusive_ptr_target, creating compile-time coupling. However, this is acceptable for TensorImpl since it is the root data structure.

**Evidence**: `c10/core/TensorImpl.h` line 1 shows `class TensorImpl : public c10::intrusive_ptr_target`. `c10/util/intrusive_ptr.h` defines the mechanism (not inspected in detail, but referenced throughout).

### 2. Immutable Device, Mutable DispatchKeySet
**Decision**: Device is immutable after TensorImpl construction; DispatchKeySet is mutable  
**Rationale**:
- Device determines where computation occurs; changing it mid-operation would violate tensor invariants
- DispatchKeySet must be mutable because inference and training modes toggle the Autograd key dynamically
- Example: `torch.no_grad()` context manager disables autograd by unsetting the Autograd key

**Evidence**: `c10/core/Device.h` defines Device as `struct` with const methods. `c10/core/TensorImpl.h` shows `key_set_` with mutating methods like `set_key_set()`.

### 3. Separate TensorBase (ownership-neutral) vs Tensor (with operator methods)
**Decision**: TensorBase is a reference handle with no auto-generated methods; Tensor adds operator methods  
**Rationale**:
- Operator signatures (from native_functions.yaml) change frequently; regenerating Tensor methods triggers massive recompiles
- TensorBase provides a stable core abstraction that doesn't depend on operator definitions
- Internal PyTorch code can depend on TensorBase and avoid recompilation cascades

**Trade-off**: Duplication of Tensor into two types. However, both are thin wrappers around TensorImpl, so the overhead is minimal.

**Evidence**: `aten/src/ATen/core/TensorBase.h` documentation (lines 69-94) explicitly states: "TensorBase aims to break up these header dependencies... TensorBase doesn't have code-generated methods."

### 4. Optional Autograd Metadata (lazy allocation)
**Decision**: autograd_meta_ is allocated lazily (nullptr unless requires_grad=True or autograd tracking is active)  
**Rationale**:
- Most tensors in inference workloads don't require gradients; allocating full autograd state for every tensor wastes memory
- Lazy allocation defers cost until actually needed
- Example: A forward pass with all tensors in no_grad() context has minimal autograd overhead

**Evidence**: `c10/core/TensorImpl.h` shows `std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_` with getter methods that check for nullptr.

## Extension Boundaries

### Public Extension Points
1. **Custom allocators**: Implement `c10::Allocator` interface and register via `SetAllocator()` calls (used by XNNPACK, oneDNN for custom memory strategies)
2. **Custom devices**: Create new `DeviceType` enum value and register kernel dispatch handlers (example: Meta device for shape inference, PrivateUse1 for custom backends)
3. **Operator registration**: Use `TORCH_LIBRARY` macros to register kernels; the dispatcher will select them based on DispatchKeySet

### Extension Constraints
- **No modification of TensorImpl structure**: Adding fields requires recompilation of all dependent code
- **No custom dispatch keys at runtime**: DispatchKey is a compile-time enum; runtime customization must reuse existing keys
- **Memory layout is fixed**: Storage byte layout is observable (size_bytes, data_ptr); changing it breaks binary compatibility

## Runtime Implications

### Initialization
- c10 initializes *first* at program startup (before ATen, torch, autograd)
- Static initializers in c10/core/ and c10/util/ run before main()
- Example: DispatchKey registration happens via `TORCH_LIBRARY` static initializers

### TensorImpl Lifecycle
1. **Creation**: Allocator::allocate() → Storage created → TensorImpl constructed with storage reference
2. **Mutation**: In-place operations (e.g., `add_()`) modify TensorImpl metadata or Storage directly
3. **Reference tracking**: Intrusive refcount increments on copy, decrements on scope exit
4. **Destruction**: When refcount reaches zero, Storage deleter is invoked, returning memory to allocator

### Thread Safety
- **TensorImpl itself is not thread-safe**: Multiple threads must not call mutating methods on the same TensorImpl without external synchronization
- **Storage is thread-safe** (intrusive_ptr operations are atomic)
- **Allocators must be thread-safe**: CPUAllocator, CUDACachingAllocator use locks for concurrent allocation requests
- Example: torch.distributed uses thread pools; each worker thread must create its own tensor views or use mutexes

### Immutability Guarantees
- **TensorImpl shape, stride, device**: Immutable after construction (verified by const accessors)
- **Storage buffer**: Mutable only via in-place operators (tracked by VersionCounter for invalidating caches)
- **DispatchKeySet**: Mutable via `set_key_set()` (used by mode guards like `torch.no_grad()`)

## Performance Implications

### Memory Overhead
- **TensorImpl base size**: ~320 bytes (on 64-bit systems) including all fields, vtable, and intrusive_ptr state
- **Storage**: ~80 bytes per Storage object; shared across views, so amortized cost depends on sharing pattern
- **Total per tensor**: ~400 bytes + data buffer size

### Performance-Critical Paths
1. **TensorImpl creation**: Must allocate and initialize storage; critical in eager execution loops
   - Optimization: Tensor pools / pre-allocation (not yet implemented in core c10)
2. **DispatchKeySet lookup**: Bitset operations (typically 1-3 CPU cycles per key check)
   - Optimization: Dispatcher caches results; see aten/src/ATen/core/dispatch/Dispatcher.h
3. **Refcount operations**: Atomic increments/decrements on shared Storage
   - Optimization: Move semantics avoid refcount operations; std::move(tensor) has zero overhead
4. **Device dispatch**: Comparing Device enum values is O(1)
   - No overhead if device is known at compile time (common case)

### Synchronization Costs
- **Allocator locks**: CPUAllocator::allocate() acquires mutex; contention under high-frequency allocation (unlikely in practice, but visible under allocation-heavy workloads)
- **Refcount atomics**: Minimal overhead with modern CPUs (XADD instruction is ~10 cycles on Skylake)

## Ownership Boundaries

### TensorImpl Field Ownership Model

| Field | Ownership | Visibility |
|---|---|---|
| storage_ | **Reference** (shared) | Private, accessed via data_ptr() |
| sizes_and_strides_ | **Owned** (per-view) | Private, accessed via sizes(), strides() |
| device_ | **Owned** | Private (immutable), accessed via device() |
| key_set_ | **Owned** | Private, accessed via key_set() |
| autograd_meta_ | **Owned** (optional) | Private, accessed via grad_fn(), set_grad_fn() |
| version_counter_ | **Reference** (shared with views) | Private, accessed via version() |
| pyobj_slot_ | **Owned** (optional) | Private, used by Python binding layer only |

**Sharing semantics**:
- Multiple TensorImpl instances can reference the same Storage (tensor views)
- Each view has independent sizes_and_strides_ (so a[0:2, :] has different strides than a)
- version_counter_ is shared across views (modifications to one trigger version increments visible to all)

## Notes and Caveats

1. **Caffe2 legacy**: c10 emerged from the Caffe2 codebase (hence "c10" = "Caffe2" + "10"); some naming and design choices reflect this history
2. **CUDA support is optional**: cuda/ subdirectory compiles only if CUDA is enabled; code can conditionally depend on it
3. **No standard library containers**: c10 avoids std::vector, std::map internally (prefers ArrayRef, hand-rolled containers) to minimize dependencies
4. **Profiler hooks**: c10/util/logging_is_not_google_glog.h provides TORCH_CHECK / TORCH_WARN macros that integrate with the profiler
