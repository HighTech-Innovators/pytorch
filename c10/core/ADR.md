# `c10/core`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/core` defines the universal tensor representation and the primitives every other subsystem builds on. It owns `TensorImpl` (the single data structure backing every tensor regardless of device, dtype, or layout), `StorageImpl` (reference-counted raw memory ownership), the `DispatchKeySet` routing bitset, the `ScalarType`/`TypeMeta` type system, `Device`/`DeviceType`, the `Allocator` interface, and the symbolic-shape types (`SymInt`, `SymBool`, `SymFloat`). Computation lives elsewhere in dispatched kernels; this directory describes where data lives and how to interpret it.

## Key Files

| File | Purpose |
|---|---|
| `TensorImpl.h` | Universal tensor metadata: storage pointer, sizes/strides, offset, dtype, device, `DispatchKeySet`, optional PyObject slot |
| `StorageImpl.h` | Reference-counted raw allocation (`DataPtr`, `numel`, `Allocator*`, resizable flag) |
| `Storage.h` | `intrusive_ptr<StorageImpl>` wrapper handle |
| `DispatchKeySet.h` | 64-bit dispatch routing bitset with `highestPriorityTypeId()` |
| `DispatchKey.h` | Dispatch key enum (backend components + functionality keys) |
| `ScalarType.h` | Data type enumeration (Float32, Int64, BFloat16, complex, quantized, bool) |
| `Scalar.h` | Tagged single-value scalar box |
| `Allocator.h` | Abstract `allocate()` / deleter memory interface |
| `CPUAllocator.h` | Aligned CPU allocation for SIMD |
| `Device.h` / `DeviceType.h` | `{DeviceType, DeviceIndex}` device identity |
| `SymInt.h` / `SymBool.h` / `SymFloat.h` | Symbolic scalars for dynamic-shape tracing |
| `impl/SizesAndStrides.h` | Small-buffer-optimized shape/stride storage |
| `impl/LocalDispatchKeySet.h` | Thread-local included/excluded key sets (`no_grad`, etc.) |
| `impl/COW.h` | Copy-on-write storage support |

## Public Interface

`TensorImpl` exposes metadata accessors (`sizes()`, `strides()`, `dim()`, `numel()`, `storage_offset()`, `dtype()`, `device()`, `key_set()`) and mutators used by kernels and views. `Storage`/`StorageImpl` expose `data_ptr()`, `nbytes()`, and allocator access. `DispatchKeySet` provides bitwise composition (`|`), exclusion, and `highestPriorityTypeId()`. `Allocator::allocate(size_t)` returns a device-tagged `DataPtr`. The `SymInt`/`SymBool` types present arithmetic operators that transparently switch between concrete integers and symbolic nodes.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/util](c10/util/ADR.md) | depends-on | `intrusive_ptr`, `ArrayRef`, `Exception`, `SmallVector`, type-list utilities |
| `c10/macros` (no ADR — excluded) | depends-on | Export macros and platform feature flags (`C10_API`, `C10_EXPORT`); EXCLUDED from ADR coverage |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depended-on-by | Dispatcher routes on `DispatchKeySet` computed from `TensorImpl` |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depended-on-by | Autograd reads/sets Autograd dispatch keys and view metadata |

## Runtime Behaviour

At tensor construction, `TensorImpl` assembles its `DispatchKeySet` from device (backend component), layout (Dense/Sparse functionality), and `requires_grad` (Autograd functionality). When an operator runs, the dispatcher ORs the key sets of all input tensors with the thread-local included set, subtracts the excluded set, and selects the highest-priority key with a single count-leading-zeros instruction. Views construct a new `TensorImpl` that shares the source `StorageImpl` (incrementing its intrusive refcount) with distinct sizes, strides, and `storage_offset_`, so no data is copied. Context managers such as `no_grad()` mutate `impl/LocalDispatchKeySet` thread-local state to exclude the Autograd key so backward graph construction is skipped.

## Performance Profile

Metadata access is on the hottest path in the framework — every operator reads sizes, strides, and device before dispatching — so `TensorImpl` is a flat struct with no virtual metadata methods, and `SizesAndStrides` inlines small shapes to avoid heap allocation. Reference counting is intrusive (`c10::intrusive_ptr`), keeping the refcount inside the object and avoiding the separate control-block allocation and second cache line that `std::shared_ptr` incurs. Dispatch key selection is O(1) via CLZ, and key-set composition/exclusion are single bitwise operations, which keeps dispatch overhead negligible relative to kernel execution.

## Design Rationale

A single `TensorImpl` for all backends avoids virtual dispatch on metadata access, which happens billions of times per training run; behavioral polymorphism is pushed into the external dispatch system instead of into subclasses. The `DispatchKeySet` bitset is chosen over virtual methods because behaviors must compose — a tensor can be simultaneously sparse, on a given backend, gradient-requiring, and under a compiler trace — and a bitset gives O(1) lookup, composition, and exclusion. Views share storage because deep-learning workloads reshape and slice constantly; copying on every reshape would make training memory-copy-bound. Symbolic scalar types exist so the compiler stack can trace shape-dependent control flow without concrete sizes.
