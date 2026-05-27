# Architecture Decision Record: C10 Core Library

**Location:** `./src/c10/core/`  
**Last Updated:** 2026-05-27  
**Classification:** Runtime Critical, State Owner, Foundation Layer

---

## Executive Summary

C10 (Caffe10) Core is the lowest-level, foundational library in PyTorch that provides type system, device abstraction, memory allocation interfaces, and dispatch infrastructure. Every other layer in PyTorch depends on c10/core. It defines the type-level contracts that all operations must satisfy.

---

## Architectural Role

C10 Core serves as the **type system and device abstraction foundation** for all of PyTorch. It establishes:

1. **Type System** — Defines the type metadata (`TypeMeta`, `ScalarType`) that every tensor carries
2. **Device Model** — Abstracts CPU, CUDA, Metal, XPU, and custom device backends
3. **Memory Allocation** — Provides the `Allocator` interface that device-specific allocators implement
4. **Dispatch Infrastructure** — Implements the dispatch key system (`DispatchKey`, `DispatchKeySet`) that routes operations to device-specific kernels
5. **Autograd State** — Provides the base interface for autograd metadata storage on tensors

Because c10/core defines these low-level abstractions, **all changes to c10/core ripple through the entire PyTorch stack**. It is the interface contract that ATen, torch/, and all device backends must satisfy.

---

## Responsibilities

### What It Does

- **Type Metadata Storage**: Stores type information for tensors via `caffe2::TypeMeta` (int32, float32, int64, bool, complex128, etc.)
- **Device Abstraction**: Defines the `Device` class and device type enumeration (CPU, CUDA, Metal, XPU, Vulkan, HIP, MPS)
- **Memory Allocators**: Provides abstract `Allocator` interface and default implementations (`CPUAllocator`, `CachingDeviceAllocator`)
- **Dispatch Keys**: Implements the dispatch table mechanism (`DispatchKey` enum, `DispatchKeySet` bitset) that selects which kernel implementation to run
- **DeviceGuard**: Context manager for device scope switching
- **Autograd State Interface**: Defines `AutogradState` and `AutogradMetaInterface` — the hooks for autograd metadata storage

### What It Does NOT Do

- **Allocate Memory** — Only defines the allocator interface; device-specific allocators (in `aten/src/ATen/native/`) implement allocation
- **Store Tensor Metadata** — Does not define tensor shape, strides, or offset (that's in `TensorImpl` in c10/core, see below)
- **Implement Kernels** — Dispatch only routes to kernels; it doesn't run operations itself
- **Execute Autograd** — Only provides metadata storage; autograd engine is in `torch/autograd/` and `torch/csrc/autograd/`

---

## Dependencies

### What Depends On C10 Core

- **ATen** (`./src/aten/`) — Uses Device, TypeMeta, Allocator, DispatchKey
- **Torch** (`./src/torch/`) — Python API uses c10/core types and dispatch infrastructure
- **All Device Backends** — CUDA allocators, Metal ops, etc. implement c10 interfaces
- **Autograd Engine** — Uses autograd metadata interfaces

### What C10 Core Depends On

- **Standard Library** — STL only; no external dependencies
- **Platform Abstractions** — May call platform-specific memory allocation APIs (mmap, malloc, etc.)
- **No higher-level layers** — c10/core has no circular dependencies

**Dependency Direction**: Unidirectional. Everything depends on c10/core; c10/core depends on nothing in PyTorch.

---

## Trade-Offs and Design Decisions

### Decision 1: Type System via Enum + Metadata Registry

**What**: `ScalarType` is an enum (int8, float32, complex64, etc.) + `TypeMeta` class that maps enum to size, byte-aligned marshaling functions.

**Why**:
- Early Decision: PyTorch predates modern C++ reflection (pre-C++17 standard reflection proposals)
- Performance: Dispatch can be done with switch/case on enum (O(1))
- Simplicity: No runtime registration needed; types are compile-time fixed

**Alternatives**:
- Runtime type registry (like Java reflection) — would require registration sites
- C++ RTTI (typeinfo) — not suitable for numeric types; would require separate type name registry

**Trade-offs**:
- **Pro**: Fast dispatch, simple code
- **Con**: Adding a new type requires changes to the enum and TypeMeta registry; not as extensible as a plugin registry

---

### Decision 2: Device as Lightweight Enum + Type

**What**: `Device` contains `type_` (enum: CPU, CUDA, Metal, etc.) and `index_` (device number, e.g., CUDA:0, CUDA:1).

**Why**:
- Multiple Devices of Same Type: A machine can have multiple GPUs or multiple custom devices
- Simple Serialization: `(type, index)` is easier to serialize than complex objects
- Early Binding: Device identity is determined at allocation time, not lazily

**Alternatives**:
- Single Global Device Context — inflexible for multi-device scenarios
- Device Objects with Methods — heavier weight; would require virtual dispatch for device-specific logic

**Trade-offs**:
- **Pro**: Lightweight, supports multi-device scenarios, easy to reason about device identity
- **Con**: Device operations (allocation, sync, etc.) are dispatched elsewhere (allocators); Device itself is passive

---

### Decision 3: Allocator as Abstract Interface

**What**: `Allocator` is a virtual interface with `allocate()` and `deallocate()` methods. Each device implements this interface.

**Why**:
- Device-Specific Strategies: CPU memory alignment differs from CUDA; CUDA uses cudaMalloc; Metal uses MTLBuffer
- Caching: CUDA backend can implement caching allocators; CPU may use different strategy
- Policy Injection: Users can provide custom allocators without modifying core code

**Alternatives**:
- Global Allocation Function — inflexible; would require device checks inside allocation logic
- Monolithic Allocator Class — would be thousands of lines with device-specific conditional logic

**Trade-offs**:
- **Pro**: Extensible; device-specific policies don't pollute core
- **Con**: Virtual method calls add indirection cost; allocation is on hot path in some workloads

---

### Decision 4: Dispatch Keys as Compile-Time Enum + Bit Flags

**What**: `DispatchKey` enum (44+ entries: CPU, CUDA, Autograd, Sparse, Quantized, Functorch, TorchScript, etc.) stored as `DispatchKeySet` bitset.

**Why**:
- Layered Dispatch: A tensor can match multiple keys (e.g., CUDA + Sparse + Quantized)
- Set Operations: Multiple keys can be combined with bitwise operations; enables batching fallback logic
- Static Registry: Keys are compile-time fixed; no runtime registration for base keys

**Alternatives**:
- Single Device-Only Key — loses ability to handle multiple layers (autograd, sparse, quantized) simultaneously
- String-Based Key Registry — slower; requires hash table; doesn't compose with bitwise operations

**Trade-offs**:
- **Pro**: Enables sophisticated dispatch (multiple layers); efficient set operations
- **Con**: Adding a new key requires compilation; keys are not user-extensible at runtime

---

## Extension Boundaries

### Extending C10 Core

**Supported Extensions:**
1. **New Device Backend** — Implement an allocator + register a DispatchKey
2. **New Scalar Type** — Add to `ScalarType` enum and `TypeMeta` registry (requires recompilation)
3. **Custom Allocators** — Subclass `Allocator` and inject via `get_allocator_for_device()`

**NOT Supported:**
- Adding dispatch keys at runtime (keys are enum, compile-time fixed)
- Modifying type system after initialization
- Changing device type enumeration

### Integration Points

- **Device Backend Integration**: Via allocator interface (CUDA backend provides `CUDAAllocator`)
- **Autograd Integration**: Via `AutogradMetaInterface` — autograd can attach metadata to tensors without c10/core knowing autograd details
- **Dispatch Integration**: Via DispatchKeySet operations — higher layers build dispatch tables using c10 keys

---

## Runtime Implications

### Initialization Order

**At Library Load:**
1. `c10/core/TypeMeta.cpp` — Initialize type registry (sizeof, alignment, copy/move functions for each ScalarType)
2. `c10/core/Device.cpp` — Initialize device type registry
3. Device-specific backends — Register allocators via `device_allocator_registry`

**At First Use:**
- Allocators are lazily instantiated when first tensor of a given device is created

### Concurrency and Thread Safety

- **Type System**: Read-only after initialization; thread-safe
- **Allocators**: Must be thread-safe; device backends (CUDA, CPU) are thread-safe
- **Dispatch Keys**: Read-only after initialization; no runtime modification
- **Device Context**: Thread-local via `DeviceGuard` — each thread can have its own "current device"

### Failure Modes

- **Invalid Device Type**: Attempting to allocate on an unsupported device throws exception
- **Allocation Failure**: Allocator returns nullptr or throws if out of memory
- **Invalid Type**: Attempting to access tensor data with wrong `ScalarType` causes incorrect interpretation (not caught)

---

## Performance Implications

### Hot Paths and Allocations

1. **Allocator Lookup** — O(1), hash table on device type; happens once per tensor creation
2. **Dispatch Key Matching** — Bitwise AND operation; O(1) per key check
3. **Type Size Lookup** — O(1) hash table on `ScalarType`; used during data access

### Known Bottlenecks

- **Allocator Indirection** — Virtual method call in hot path during tensor creation; address space randomization (ASLR) can cause cache misses
- **Multi-Device Overhead** — Checking device context on every operation has measurable cost in microbenchmarks

### Mitigation Strategies

- **Default Device Cache** — Common case (CPU or single GPU) caches device allocator
- **Dispatch Table Monomorphization** — Generated code specializes dispatch for common key combinations

---

## Ownership Boundaries

### What C10 Core Owns

- Type system and type metadata
- Device abstraction and registry
- Allocator interface and default CPU implementation
- Dispatch key enumeration and key set operations
- Autograd metadata storage hooks

### What C10 Core Does NOT Own

- Device-specific allocators (owned by device backends; c10 only defines interface)
- Dispatch table (owned by operator registration system in aten/)
- Kernel implementations (owned by backend-specific folders)
- Autograd state (owned by torch/autograd/)

### State Shared with Other Layers

- **Type Registry** — Read by ATen, torch/, all kernels
- **Allocators** — Owned by c10 but registered by device backends
- **Dispatch Keys** — Set by tensor creation, used by dispatch system

---

## Testing and Validation

### Critical Tests

- **Type System**: Verify `sizeof()` and alignment correctness for all scalar types
- **Device Creation**: Verify valid device types, device index ranges
- **Allocator Dispatch**: Verify correct allocator is selected for each device type
- **Dispatch Keys**: Verify bitset operations and key combinations

### Known Gaps

- No explicit concurrency tests for allocator thread-safety under contention
- Limited testing of custom allocator injection

---

## Related Systems

- **ATen** — Primary consumer; implements kernels that allocate via c10 allocators
- **torch/autograd/** — Uses autograd metadata interface from c10
- **Operator Dispatch** — Uses DispatchKeySet for kernel routing
- **Device Backends** — Implement allocators and register dispatch keys

---

## References

- `c10/core/TensorImpl.h` — Tensor metadata storage (uses c10 types)
- `c10/core/Storage.h` — Storage with allocator (uses c10 Device, Allocator)
- `c10/core/TypeMeta.cpp` — Type system initialization
- `c10/core/Device.cpp` — Device type definitions
- `c10/core/DispatchKey.cpp`, `DispatchKeySet.cpp` — Dispatch infrastructure
- Book Chapter 4 (Tensor Core) — Explains TensorImpl and how it uses c10 types
- Book Chapter 3 (Operator Dispatch) — Explains how DispatchKey is used for kernel selection
