# Architecture Decision Record: c10/core

## Architectural Role

`c10/core` is the foundational tensor metadata layer that defines the internal representation of all PyTorch tensors. It provides the essential data structures (`TensorImpl`, `StorageImpl`, `DispatchKeySet`) that enable every tensor operation in the system. This module is Runtime Critical because every computation—from simple element-wise operations to distributed backward passes—fundamentally depends on these abstractions. The module owns tensor shape, dtype, device, memory layout, and dispatch routing information.

## Responsibilities

This module is responsible for:
- Defining `TensorImpl` (the metadata backbone for all tensors with ~3381 lines defining fields for sizes/strides, dtype, device, storage reference, and autograd metadata)
- Managing `StorageImpl` for memory ownership with reference-counted lifecycle and support for views via `storage_offset_` and shape transformations
- Implementing `DispatchKeySet` (64-bit routing bitset combining backend keys like CPU/CUDA and functionality keys like Autograd/Sparse)
- Providing intrusive pointer utilities (`intrusive_ptr_target`, `IntrusivePtr.h`) for thread-safe reference counting
- Defining allocator abstractions (`Allocator.h`) for device-specific memory strategies (CPU malloc, CUDA caching allocator, MPS Metal)
- Supporting tensor views through shape/stride metadata without data copy
- Tracking tensor mutation via `version_counter_` for backward correctness

The module does **not** implement actual computation kernels, operator dispatch logic (that belongs to `aten/src/ATen/core/dispatch`), or distributed communication.

## Dependencies

**Inbound** (what depends on c10/core):
- Nearly every component in the system: `aten/src/ATen/core`, `aten/src/ATen/native`, `torch/csrc/autograd`, `torch/nn`, `torch/distributed`
- External: Standard library (STL containers, smart pointers), system allocators

**Outbound** (what c10/core depends on):
- `c10/util` for exception utilities and intrusive pointers (inverse dependency: c10/core depends on c10/util's foundations)
- System libraries for memory allocation (posix_memalign, pthread, etc.)
- No dependency on ATen or higher-level PyTorch code

## Trade-offs

**Immutable metadata after construction**: TensorImpl fields like `device_` and `data_type_` are set during construction and cannot change. This prevents runtime device migration but ensures consistency and thread-safety without locks.

**Optional autograd_meta_ for memory efficiency**: Tensors that don't require gradients (common in inference) have `autograd_meta_ = nullptr`, saving heap allocations. The cost is conditional null-checks during gradient tracking. This design trades code complexity for memory efficiency.

**Reference counting over garbage collection**: StorageImpl uses intrusive_ptr for deterministic memory lifecycle. This requires explicit ptr management but enables predictable deallocation and avoids GC pauses during inference.

**64-bit DispatchKeySet bitset**: Single 64-bit integer limits dispatch features to 64 distinct keys (backend + functionality combinations). This is sufficient for current use but prevents unlimited extensibility. The alternative (hashmap) would be slower on every operator call.

**Symbolic shapes (SymInt) for compilation**: Dimensions can be symbolic expressions (e.g., "batch_size") instead of concrete integers, enabling torch.compile to generate polymorphic kernels. The trade-off is added complexity in shape operations and potential overheads during runtime checks.

## Extension Boundaries

- **Device extensibility via PrivateUse1 backend**: Third-party hardware (custom ASICs, emerging accelerators) can use the `PrivateUse1` dispatch key (reserved in `DispatchKeySet`) to register custom kernels without modifying c10/core.
- **Custom allocator registration**: New devices can implement the `Allocator` interface and register it at startup. Existing tensors transition to new memory schemes transparently.
- **Autograd integration point**: `AutogradMetaInterface` (pointer in TensorImpl) allows alternative autograd backends to attach gradient information without modifying TensorImpl itself.

## Runtime Implications

**Lifecycle**: TensorImpl is allocated once per tensor creation and lives until the tensor is destroyed. StorageImpl is reference-counted: when all TensorImpls referencing it release their pointers (via intrusive_ptr destructor), the allocator is invoked to free the memory.

**Initialization order**: TensorImpl requires valid device, dtype, and allocator before construction; there are no lazy fields.

**Concurrency**: TensorImpl and StorageImpl are immutable after construction (except for `version_counter_` which has atomic mutation). This enables read concurrency without locks. Mutation of tensor data (via in-place operations) is protected by the version counter; the system raises an error if a tensor is modified after being saved for backward during gradient computation.

**Inference mode**: In inference, `version_counter_` is disabled (bumping becomes no-op), reducing per-mutation overhead.

## Performance Implications

**Cache efficiency**: TensorImpl layout is optimized for cache lines; the actual struct size is tuned (typically 208-256 bytes on 64-bit systems) to fit multiple instances in L1 cache for vectorized operations.

**Allocation overhead**: StorageImpl allocations are pooled at the allocator level (especially for CUDA via CUDACachingAllocator) to amortize system call costs.

**Shape computation**: `sizes_and_strides_` uses inline storage for ≤5 dimensions (no heap allocation); higher-rank tensors incur one extra pointer dereference.

**View overhead**: Creating views is O(1) — only TensorImpl metadata is copied; the underlying StorageImpl is reference-counted and shared. No data copy occurs.

## Ownership Boundaries

- **TensorImpl owns**: sizes, strides, dtype, device, storage offset, dispatch keys, autograd metadata pointer (if present)
- **TensorImpl borrows**: StorageImpl (via intrusive_ptr; SharedPtr semantics)
- **StorageImpl owns**: raw memory buffer, allocation size, reference to allocator
- **Allocator owns**: no data; it is called by StorageImpl to manage lifetime

## Verification Points

- `c10/core/TensorImpl.h:2943-3001` — Storage/TensorImpl integration and ownership
- `c10/core/DispatchKey.h:1-50` — Dispatch key enumeration
- `c10/core/DispatchKeySet.h` — Dispatch routing bitset
- `c10/core/impl/SizesAndStrides.h` — Inline shape storage for efficiency
