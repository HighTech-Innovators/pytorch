# `c10/util`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/util` provides the low-level C++ utility layer that the rest of PyTorch builds on: intrusive reference counting, the exception/assertion system, thread-local storage, small-buffer-optimized containers, half-precision types, logging, and a type-registry. It has no PyTorch-specific concepts — it is pure infrastructure with no upward dependencies.

## Key Files

| File | Purpose |
|---|---|
| `c10/util/intrusive_ptr.h` | `intrusive_ptr<T>` / `weak_intrusive_ptr<T>` / `intrusive_ptr_target` — the smart pointer behind every tensor |
| `c10/util/Exception.h` | `c10::Error`, `TORCH_CHECK`, `TORCH_INTERNAL_ASSERT`, `C10_THROW_ERROR` |
| `c10/util/Exception.cpp` | Error message formatting and stack-trace capture |
| `c10/util/ThreadLocal.h` | Portable thread-local storage wrapper |
| `c10/util/SmallVector.h` | Small-buffer-optimized vector (avoids heap for short sequences) |
| `c10/util/ArrayRef.h` | Non-owning view over contiguous data (used for sizes/strides) |
| `c10/util/irange.h` | Range helper for bounds-safe integer loops |
| `c10/util/Half.h` | `Half` / bfloat16 scalar types and conversions |
| `c10/util/Logging.h` | Logging facade over glog or the built-in fallback |
| `c10/util/typeid.h` | `TypeMeta` runtime type metadata |

## Public Interface

`intrusive_ptr<T>`, `weak_intrusive_ptr<T>`, `intrusive_ptr_target`, `make_intrusive<T>()`, `Error`, `TORCH_CHECK`, `TORCH_INTERNAL_ASSERT`, `SmallVector<T, N>`, `ArrayRef<T>`, `irange()`, `Half`, `optional<T>`, `TypeMeta`, `Registry`, `ThreadLocalDebugInfo`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| Standard C++ library only | depends-on | No PyTorch dependencies; leaf of the dependency graph |
| [c10/core](c10/core/ADR.md) | depended-on-by | `TensorImpl`, `Storage`, `DispatchKeySet` all use these utilities |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen headers pervasively use `ArrayRef`, `SmallVector`, `TORCH_CHECK` |

## Runtime Behaviour

`intrusive_ptr` stores the reference count inside the pointed-to object (`intrusive_ptr_target`) rather than in a separate control block, and `c10/util/intrusive_ptr.h` packs the strong and weak counts into a single `combined_refcount_` atomic (`atomic_combined_refcount_increment`/`_decrement`), so a copy is one atomic add and a destroy is one atomic subtract plus a zero-check. `TORCH_CHECK` and `TORCH_INTERNAL_ASSERT` expand (via `C10_THROW_ERROR`) to throw a `c10::Error` carrying a formatted message and, optionally, a captured C++ stack trace. `ThreadLocal` storage backs per-thread debug info and dispatcher state.

## Performance Profile

`intrusive_ptr`'s combined-refcount atomic is the single most frequently executed synchronization primitive in the framework — every tensor copy touches it — so it is engineered to be one atomic RMW rather than two. `SmallVector` and `ArrayRef` avoid heap allocation for the short size/stride sequences that dominate tensor metadata, keeping tensor construction allocation-free in the common case. `TORCH_CHECK` is structured so the success path is a cheap predicate test; message formatting and stack capture happen only on the failure branch, keeping assertions off the hot path.

## Design Rationale

Intrusive reference counting (count stored in the object) avoids the separate control-block allocation and extra indirection of `std::shared_ptr`, which matters because tensors are created and destroyed constantly. Small-buffer-optimized containers exist because tensor ranks are almost always tiny, so heap allocation for sizes/strides would be pure overhead. The exception macros centralize error reporting so operator code can assert cheaply and consistently. Keeping `c10/util` free of PyTorch concepts makes it a stable, reusable leaf that anchors the strictly-upward dependency direction of the whole codebase.
