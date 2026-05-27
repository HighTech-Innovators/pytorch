# Architecture Decision Record: c10 Core Tensor Library

## Architectural Role

**c10** is the foundational C++ library providing hardware-agnostic tensor primitives that all higher PyTorch layers depend upon. It provides the minimal type system and memory management infrastructure required to represent tensors across different hardware backends (CPU, CUDA, XPU, MPS, etc.) without any device-specific computation logic.

**Location**: `c10/` | **Language**: C++17 | **Dependencies**: None (foundation)

## Responsibilities

**c10 owns**:
- Core tensor representation (`TensorImpl`, `Storage`, `StorageImpl`) 
- Memory allocation abstraction (`Allocator` interface) and device abstraction (`Device`, `DeviceType`)
- Dispatch key metadata (`DispatchKeySet`) that controls kernel routing
- Intrusive pointer and reference counting infrastructure
- Utility types (ArrayRef, TypeList, Exception hierarchy, etc.)
- Device-specific utilities for CUDA, HIP, XPU, Metal (no kernels)

**c10 does not own**:
- Tensor computation kernels (ATen owns those)
- Autograd graph construction (autograd owns that)
- Python bindings (torch/csrc owns that)
- Backend-specific implementations beyond device abstraction

## Dependencies

### Inbound Dependencies
- **ATen** imports c10 core types for kernel implementation (TensorImpl, StorageImpl, DispatchKey)
- **autograd** imports c10 for tensor representation and dispatch mechanisms
- **torch.nn** and Python modules import c10 types through ATen and autograd
- **All backends** (CPU, CUDA, HIP, XPU, Metal) use c10's Device abstraction

### Outbound Dependencies
- **None** — c10 is the foundation layer with no dependencies on higher layers

## Trade-offs and Design Decisions

### 1. Storage Separated from TensorImpl
**Decision**: A `TensorImpl` does not own raw memory directly. It holds a `Storage` object (reference-counted handle to `StorageImpl`).

**Rationale**: Enables multiple tensors to share the same underlying buffer (views, slices) without deep copying. This is essential for memory efficiency in tensor operations.

**Alternative considered**: Embedding storage directly in TensorImpl would simplify the ownership model but prevent efficient view semantics.

### 2. SizesAndStrides Heap-Allocated for High-Dimensional Tensors
**Decision**: For tensors with more than ~5 dimensions, `sizes_and_strides_` is a `unique_ptr` to a heap structure; otherwise inlined.

**Rationale**: Avoids stack bloat for common cases (2D-4D tensors) while supporting arbitrary dimensionality.

**Evidence**: Source shows optional `SizesAndStrides` structure with inlining optimization.

### 3. DispatchKeySet for Runtime Dispatch Control
**Decision**: Each tensor carries a bitset (`DispatchKeySet`) indicating which dispatch keys apply (e.g., CPU, AutogradCPU, Python).

**Rationale**: Enables PyTorch to support multiple backends and extensible behaviors (autograd, profiling, tracing) without branching on tensor type. The dispatcher uses this set to select which kernel implementation to call.

**Trade-off**: Small per-tensor overhead (bitset storage) but massive flexibility in routing and extensibility.

### 4. Optional Device Field
**Decision**: Device is stored as `std::optional<Device>` rather than always-present.

**Rationale**: In some construction paths, device can be inferred from context or parent tensor. Immutable once set for a TensorImpl's lifetime.

### 5. ScalarType for Element Representation
**Decision**: `ScalarType` enum (kFloat, kInt64, kBool, etc.) determines element size and representation, not computation semantics.

**Rationale**: c10 is computation-agnostic. It knows how many bytes one element occupies but doesn't know what a float means operationally — that's for ATen kernels.

## Extension Boundaries

**Device support**: New backends integrate via:
- Adding DeviceType enum entry
- Implementing `Allocator` interface for memory management
- Creating backend-specific utilities in `c10/<backend>/` subdirectory
- Registering in dispatcher (ATen's responsibility, not c10's)

**Custom scalar types**: New scalar types can be registered via ScalarType enum, but c10 provides no mechanism to define custom tensors with different internal structure. Such extensions require changes to TensorImpl itself or wrapper types at higher layers.

**Dispatch extension**: New behaviors (autograd, tracing, profiling) are integrated by adding DispatchKey entries and registering kernels at higher layers (ATen/autograd); c10 provides only the mechanism.

## Runtime Implications

### Initialization
- c10 requires minimal initialization — mostly static registration of scalar types and device types
- No threads or side effects during library load

### Memory Lifecycle
- `TensorImpl` and `Storage` use reference counting (`intrusive_ptr`) to manage lifetime
- Memory is deallocated when reference count reaches zero; final deallocation delegated to `Allocator`
- No garbage collection or cycle detection — caller responsible for breaking reference cycles

### Concurrency
- `TensorImpl` is **not thread-safe** for mutations after creation
- Multiple threads can safely read immutable TensorImpl (e.g., shape, dtype, device) after construction
- `StorageImpl` uses atomic reference count; safe for concurrent reference counting but not for mutations to data

### Lifecycle and Initialization Order
- c10 library has no required initialization beyond standard C++ static construction
- Device types are registered at static initialization time
- Allocators are registered lazily or at startup depending on backend (e.g., CUDA allocator registered when CUDA is first used)

## Performance Implications

### Known Hotspots
1. **TensorImpl access**: Shape and stride lookups (`size()`, `stride()`) are inline methods on `TensorImpl` — very fast O(1)
2. **DispatchKeySet membership test**: Bitset lookup to determine applicable dispatch keys — O(1) bitwise operation
3. **Storage ref-counting**: Increment/decrement during tensor creation/destruction — atomic operations, modest cost
4. **Allocator dispatch**: Memory allocation delegates to backend-specific allocator; cost depends on backend (CPU malloc vs CUDA malloc)

### Allocation Patterns
- Small tensors (< 1MB typical): Allocated individually via backend allocator
- Large tensors: Backend-specific pooling/caching may apply (e.g., CUDA caching allocator)
- Views: No new allocation — new TensorImpl created with reference to same Storage

### Synchronization Costs
- No explicit synchronization primitives in c10 — backends handle device synchronization
- Reference counting uses atomic operations (modest overhead, typically < 1% of tensor operation cost)

## Ownership Boundaries

**c10 owns**:
- Representation of tensor metadata (shape, stride, dtype, device, dispatch keys)
- Memory handle (Storage) and allocation abstraction, not raw memory
- Reference counting and lifetime management
- Type and device abstractions

**c10 borrows or delegates**:
- Raw memory contents (delegated to backends via Allocator)
- Element interpretation (delegated to kernels via ScalarType)
- Computation logic (owned by ATen)
- Device lifecycle (owned by backends, not c10)

**Parent layers own**:
- Autograd graph (autograd engine owns)
- Python wrapper objects (torch/csrc owns)
- Kernel implementations (ATen owns)
