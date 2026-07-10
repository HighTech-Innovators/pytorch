# `c10/core`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/core` owns the core tensor runtime state (`TensorImpl`), the storage abstraction (`Storage`/`StorageImpl`), the memory-allocation interface (`Allocator`/`DataPtr`), the device model (`Device`/`DeviceType`), the scalar-type registry (`ScalarType`), and the dispatch-key registry (`DispatchKey`/`DispatchKeySet`) that selects which kernel runs for every operator.

## Key Files

| File | Purpose |
|---|---|
| `c10/core/TensorImpl.h` | Tensor metadata: `storage_`, `sizes_and_strides_`, `numel_`, `data_type_`, `key_set_`, `autograd_meta_` |
| `c10/core/TensorImpl.cpp` | TensorImpl construction, resize, and version-counter logic |
| `c10/core/StorageImpl.h` | Owns a `DataPtr`, byte count, and allocator; the buffer behind tensors |
| `c10/core/Storage.h` | Thin `intrusive_ptr<StorageImpl>` handle |
| `c10/core/DispatchKey.h` | Enumeration of backend and functionality dispatch keys |
| `c10/core/DispatchKeySet.h` | `uint64_t` bitset with `highestPriorityTypeId()` selection |
| `c10/core/Allocator.h` | Abstract `Allocator` interface, `DataPtr`, `SetAllocator`/`GetAllocator` |
| `c10/core/CPUAllocator.cpp` | Cache-aligned CPU allocator implementation |
| `c10/core/Device.h` | Device type + index (`cpu`, `cuda:0`, …) |
| `c10/core/ScalarType.h` | dtype enumeration (float32, int64, bfloat16, …) |
| `c10/core/SymInt.h` | Symbolic integer for dynamic-shape (`torch.compile`) support |

## Public Interface

`TensorImpl`, `UndefinedTensorImpl`, `Storage`, `StorageImpl`, `Allocator`, `DataPtr`, `SetAllocator()`, `GetAllocator()`, `Device`, `DeviceType`, `ScalarType`, `Scalar`, `DispatchKey`, `DispatchKeySet`, `DispatchKeySet::highestPriorityTypeId()`, `SymInt`, `Layout`, `MemoryFormat`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/util](c10/util/ADR.md) | depends-on | `intrusive_ptr`, `Exception`/`TORCH_CHECK`, `SmallVector`, `ArrayRef`, `typeid` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen operators read `TensorImpl` state and dispatch on `DispatchKeySet` |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depended-on-by | Autograd stores `AutogradMeta` via `TensorImpl::autograd_meta_` |

## Runtime Behaviour

Every PyTorch tensor is a handle wrapping `c10::intrusive_ptr<TensorImpl>`; `TensorImpl` holds `storage_` as a direct member while `autograd_meta_` is a lazily-allocated unique pointer, so autograd state costs nothing until `requires_grad=True`. A view operation (e.g. `transpose`) allocates a new `TensorImpl` that shares the same `StorageImpl` with different strides and offset, and in-place mutation bumps a version counter so autograd can detect stale saved tensors. Allocators are registered once at startup via `SetAllocator(DeviceType, Allocator*)` and retrieved per allocation with `GetAllocator(DeviceType)`.

## Performance Profile

`DispatchKeySet` is a `uint64_t` bitmask so set union, masking, and `highestPriorityTypeId()` are branch-light bit operations on the per-operator hot path. Tensor allocation crosses `Allocator::allocate()` — the CPU allocator in `CPUAllocator.cpp` uses aligned allocation, and every dynamic-shape workload repeatedly pays this cost. Reference-count churn on `TensorImpl` is an atomic operation, so creating and destroying many small temporaries in a loop is measurably more expensive than reusing tensors. Because views share storage, transpose/slice are O(1) metadata operations that avoid data movement entirely.

## Design Rationale

Separating `TensorImpl` from `Storage` makes views cheap: metadata can change without moving bytes, at the cost of in-place mutations being visible through all views (mitigated by version counters). Using a `DispatchKey` table instead of virtual methods keeps `TensorImpl`'s layout small and lets 2000+ operators and new backends register without touching the core class. The pImpl handle (`Tensor` → `intrusive_ptr<TensorImpl>`) gives ABI stability so downstream C++ consumers need not recompile when internals change. CUDA-flavoured dispatch keys exist in the enum but are inert in this CPU-only deployment.
