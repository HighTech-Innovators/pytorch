# Architecture Decision Record: c10 (Core Abstractions)

## Architectural Role

The `c10` subsystem provides the **foundational abstractions and C++ utilities** that underpin the entire PyTorch runtime. It defines:

1. **Core data types**: Tensor, Device, ScalarType, Storage
2. **Type system and device abstraction**: Device types (CPU, CUDA, etc.), memory management
3. **Dispatch mechanism**: How operations are routed to appropriate implementations
4. **Memory and allocator protocols**: Allocation strategies and lifecycle management
5. **Utility infrastructure**: Intrusive pointers, small vector, threading utilities

Key insight: `c10` is the **"kernel" of PyTorch's C++ runtime**. Everything else (ATen, autograd, Python bindings) depends on abstractions defined here. It is intentionally decoupled from domain knowledge (no neural network concepts, no autograd specifics).

## Responsibilities

### What This Subsystem Owns

1. **Core Types** (`c10/core/*.h`)
   - `Tensor` and `TensorImpl`: The tensor data structure and its C++ representation
   - `Device`: Abstract device type with attributes (type, index)
   - `ScalarType`: Enum for data types (float32, int64, complex128, etc.)
   - `Storage`: Memory holder with reference counting
   - `DataPtr`: Raw pointer with associated deleter

2. **Device Abstraction** (`c10/core/Device.h`, `c10/device/Device.cpp`)
   - Device type enumeration: CPU, CUDA, HIP, Metal, XPU, etc.
   - Device index: which GPU (0, 1, ...) or other accelerator
   - Device properties: query capabilities (max block size for CUDA, etc.)
   - Device management: set default device, track current device

3. **Allocator Protocol and Implementations** (`c10/core/Allocator.h`, `c10/core/CPUAllocator.cpp`)
   - Abstract `Allocator` interface: `allocate()`, `deallocate()`, `resize()`
   - CPU allocator: malloc/free with caching strategy
   - CUDA allocator: CUDAMallocCachingAllocator for GPU memory
   - Memory pool abstraction: reuse freed blocks to reduce allocation overhead

4. **Dispatch System Core** (`c10/core/Dispatcher.h`, `c10/core/OperatorSchema.h`)
   - Operator registry: map of operation names to implementations
   - Dispatch keys: CPU, CUDA, Autograd, Quantized, etc.
   - Kernel selection: given operation name and dispatch keys, find the implementation
   - Backend fallback: default implementations for unsupported backends

5. **Type Dispatch Utilities** (`c10/util/TypeList.h`, `c10/util/Tuple.h`)
   - Template metaprogramming utilities for compile-time type handling
   - Type traits: identifying types at compile time
   - Enabling/disabling code based on type properties

6. **Smart Pointers and Memory Utilities** (`c10/util/intrusive_ptr.h`, `c10/util/SmallVector.h`)
   - `intrusive_ptr`: Reference counting pointer for objects with embedded refcount
   - `SmallVector`: Optimized vector for small sizes (avoid heap allocation)
   - `Optional`: Optional type wrapping
   - Exception handling utilities

7. **Device-Specific Code** (`c10/cuda/`, `c10/hip/`, `c10/metal/`, etc.)
   - Device-specific initialization and utilities
   - CUDA: context management, stream handling
   - HIP: AMD GPU equivalents
   - Metal: iOS/macOS GPU support

### What This Subsystem Does NOT Own

- **Operations and Kernels**: ATen defines operations and provides kernel implementations
- **Automatic Differentiation**: torch.autograd defines computation graphs
- **Neural Network Abstractions**: torch.nn defines layers
- **Optimization**: torch.optim defines optimizers
- **Python Bindings**: torch.csrc and pybind11 define Python-C++ interface

## Dependencies

### Upstream Dependencies (What Uses This)

- **ATen** (`aten/`): Uses Device, Tensor, Allocator, Dispatch for all operations
- **torch.autograd**: Uses TensorImpl, Device, Dispatch
- **torch.nn**: Uses Tensor and Device indirectly through ATen/autograd
- **torch.distributed**: Uses Device for process-local device management
- **torch._C** (Python bindings): Uses all core types to interface with Python

### Downstream Dependencies (What This Uses)

- **System Libraries**: malloc, CUDA SDK, HIP SDK, Metal Framework
- **Boost** (minimal dependency): Some utilities for compatibility

### Dependency Direction

```
All PyTorch subsystems
    ↓
c10 (Core Abstractions)
    ↓
System Memory Allocators (malloc, CUDA, HIP)
```

c10 is **lowest in the dependency hierarchy**, ensuring all other subsystems can rely on stable abstractions.

## Trade-offs and Design Decisions

### Tensor as Value Type vs. Handle

**Decision**: In C++, `at::Tensor` behaves like a value type (copyable, assignable) but internally uses reference-counted storage.

**Trade-off**:
- ✅ **Advantage**: Intuitive API for users; copy semantics feel natural
- ✅ **Advantage**: Python users expect value-like behavior
- ❌ **Disadvantage**: Hidden reference counting; not obvious that copies are shallow
- ❌ **Disadvantage**: Performance implications of refcount increments on each copy

**Evidence**: `c10/core/TensorImpl.h` uses intrusive_ptr; `Tensor` copy increments refcount.

### Device as Immutable Identifier

**Decision**: `Device` (like (DeviceType::CUDA, 0)) is immutable; operations don't migrate tensors between devices transparently.

**Trade-off**:
- ✅ **Advantage**: Explicit about device placement; no hidden transfers
- ✅ **Advantage**: Easy to reason about memory locality
- ❌ **Disadvantage**: User must explicitly call `.to(device)` to move tensors
- ❌ **Disadvantage**: Multi-device operations require explicit device specification

**Evidence**: `c10/core/Device.h` has no methods to change device; immutable.

### Allocation Strategy: Caching

**Decision**: CPU and GPU allocators cache freed blocks to reuse rather than immediately deallocate.

**Trade-off**:
- ✅ **Advantage**: Faster allocation (reuse freed block vs. system malloc)
- ✅ **Advantage**: Reduced fragmentation
- ❌ **Disadvantage**: Memory not returned to OS; can appear as leak
- ❌ **Disadvantage**: Cache overhead; extra memory held

**Evidence**: `CPUAllocator`, `CudaMallocCachingAllocator` implement block caching.

### ScalarType as Enum

**Decision**: Data types are a compile-time enum, not runtime polymorphism.

**Trade-off**:
- ✅ **Advantage**: Dispatch at compile time when possible; zero overhead
- ✅ **Advantage**: Easy to add new types without modifying base classes
- ❌ **Disadvantage**: Operations must explicitly handle each type (or use template code generation)
- ❌ **Disadvantage**: Type information only available at runtime via enum value

**Evidence**: `c10/core/ScalarType.h` defines ScalarType enum; ATen uses this for dispatch.

### Operator Schema as Metadata

**Decision**: Operations define a schema (name, arguments, types, returns) that is registered and discoverable.

**Trade-off**:
- ✅ **Advantage**: Runtime introspection: can query what operations exist and their signatures
- ✅ **Advantage**: Enables automatic operator export (TorchScript, ONNX)
- ❌ **Disadvantage**: Schema must be maintained alongside implementation
- ❌ **Disadvantage**: Schema mismatches can cause hard-to-debug errors

**Evidence**: `c10/core/OperatorSchema.h`, torchgen auto-generates schemas.

## Extension Boundaries

### Adding New Devices

**Boundary**: Implement device-specific code in `c10/<device>/` directory.

1. Add DeviceType enum value
2. Implement device-specific allocator
3. Register allocator with device type
4. Add any device-specific initialization code

**Evidence**: `c10/cuda/`, `c10/hip/`, `c10/metal/` follow this pattern.

### Registering New Operators

**Boundary**: Define operator schema and implementations; register with dispatcher.

This happens in ATen/torchgen layer, not c10 directly, but c10 provides the dispatcher infrastructure.

**Evidence**: `c10/core/Dispatcher.h` provides the registry.

### Custom Memory Allocators

**Boundary**: Implement `Allocator` interface for custom allocation strategies.

```cpp
class MyAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t nbytes) override {
    // Custom allocation logic
  }
  void deallocate(c10::DataPtr ptr) override {
    // Custom deallocation logic
  }
};

// Register
c10::SetAllocator(DeviceType::CPU, std::make_unique<MyAllocator>());
```

**Evidence**: `c10/core/Allocator.h` defines interface; implementation in ATen.

## Runtime Implications

### Lifecycle and Initialization

1. **Process Start**: c10 initializes default allocators for CPU, registered device types
2. **Device Selection**: Code specifies device (CPU, CUDA:0, etc.)
3. **Allocation**: Requested operations allocate tensors via device allocator
4. **Reference Counting**: Tensor copies increment refcount; destruction decrements
5. **Process End**: Allocators destructed; cached memory released to OS

### Concurrency Behavior

**Thread Safety**:
- **Per-Device Thread Safety**: Each device has thread-local state (e.g., default stream for CUDA)
- **Allocator Thread Safety**: Allocators use locks to protect cached block lists
- **Reference Counting**: Atomic operations for refcount updates

**Evidence**: `CudaMallocCachingAllocator` uses locking; see `THCCachingAllocator_malloc`.

### Failure Behavior

1. **Allocation Failure**: Allocator raises or returns nullptr; operation fails
2. **Invalid Device**: Using uninitialized device raises error
3. **Out of Memory**: Allocator may cache eviction; eventually raises OOM error

**Evidence**: `DataPtr` can be null; operations check before use.

## Performance Implications

### Known Hotspots

1. **Refcount Increment/Decrement**: Atomic operations on tensor copy
2. **Dispatch Lookup**: Hash table lookup for operator implementation
3. **Allocator Lock Contention**: Multiple threads allocating simultaneously

### Allocation Patterns

- **Stack Allocation**: TensorImpl and metadata allocated once per tensor
- **Heap Allocation**: Raw data buffer allocated from allocator; cached for reuse
- **SmallVector Optimization**: Shape/strides stored inline for small tensors (no heap allocation)

### Synchronization Costs

- **Allocator Synchronization**: Lock held during allocation; released immediately
- **No Tensor Synchronization**: Tensors themselves are not guarded by locks

## Ownership Boundaries

### State Owned by c10

1. **Device Registry**: Available device types and properties
2. **Allocator Registry**: Allocators for each device type
3. **Operator Dispatcher**: Registry of operations and their implementations
4. **TensorImpl Metadata**: Shape, strides, dtype, device, autograd metadata

### State Borrowed from System

1. **Raw Memory**: Actual allocated bytes; c10 doesn't own the bytes, only the pointer
2. **Device Hardware**: Device properties discovered from hardware

### State Owned by Higher Layers

1. **Operator Implementations**: ATen provides concrete kernels
2. **Dispatch Rules**: ATen/torchgen defines dispatch precedence

## Key Implementation Files

| File | Purpose |
|---|---|
| `core/Tensor.h` | Tensor class and basic operations |
| `core/TensorImpl.h` | TensorImpl data structure |
| `core/Device.h` | Device abstraction |
| `core/ScalarType.h` | Data type enumeration |
| `core/Storage.h` | Memory storage wrapper |
| `core/Allocator.h` | Allocator interface |
| `core/Dispatcher.h` | Operator dispatcher |
| `core/CPUAllocator.cpp` | CPU memory allocator |
| `cuda/CUDAAllocator.cpp` | CUDA memory allocator |
| `util/intrusive_ptr.h` | Reference counting utilities |
| `util/SmallVector.h` | Optimized vector implementation |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/c10/` headers and implementation — checked for core abstractions, device types, allocator interface
2. **Book Chapter**: Chapter 03 "Tensor: Data Structure and Ownership" provides architectural understanding of TensorImpl
3. **Code Flow**: Traced from tensor creation through device selection to memory allocation

Last Verified: 2026-05-27
