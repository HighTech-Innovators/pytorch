# `c10/util`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/util` supplies the low-level C++ utility layer that `c10/core`, ATen, and dispatcher code include on hot paths. It owns the non-owning array views, intrusive reference counting, small-buffer containers, error and warning machinery, type lists, hashing containers, integer ranges, and scalar helper types that keep the core tensor stack independent of heavier framework code. Book Chapter 02 relies on this directory when it explains intrusive storage lifetime, `ArrayRef` shape arguments, and `TORCH_CHECK` diagnostics.

## Key Files

| File | Purpose |
|---|---|
| `ArrayRef.h` | Non-owning contiguous array reference with checked `front`, `back`, `slice`, and `at` accessors |
| `Exception.h` | `c10::Error`, warning classes, warning handlers, and `TORCH_CHECK`-backed diagnostics |
| `intrusive_ptr.h` | Intrusive strong and weak reference counting for `TensorImpl`, `StorageImpl`, and other c10 objects |
| `SmallVector.h` | Inline-storage vector derived from LLVM's `SmallVector` design for small metadata buffers |
| `Backtrace.h` | Lazy backtrace handle and backtrace capture entry points used by `c10::Error` |
| `Bitset.h` | Fixed-size bitset with `for_each_set_bit` and first-set-bit support for dispatcher helpers |
| `DimVector.h` | Size and stride vectors built on `SmallVector` with the tensor metadata inline capacity |
| `irange.h` | Lightweight integer range helper used in tensor loops and validation code |
| `flat_hash_map.h` | Hash-table implementation used by mobile allocators and dispatcher support code |

## Public Interface

The public interface is header-first: downstream code includes `c10/util/ArrayRef.h`, `c10/util/SmallVector.h`, `c10/util/intrusive_ptr.h`, `c10/util/Exception.h`, and related headers directly. `ArrayRef<T>` presents pointer-plus-length views over existing buffers and deliberately does not own the referenced memory. `intrusive_ptr<T>` requires `T` to inherit from `intrusive_ptr_target` and exposes strong and weak lifetime operations without a separate control block. `Error`, `Warning`, warning handlers, and macros built on them provide framework-wide diagnostics with source locations and optional backtraces.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| c10/macros | depends-on | Export macros, platform attributes, and branch prediction annotations used throughout the utility headers |
| torch/headeronly/util | depends-on | Header-only `ArrayRef` base used to share constexpr array-view operations |
| [c10/core](c10/core/ADR.md) | depended-on-by | `TensorImpl`, `StorageImpl`, `DispatchKeySet`, `Device`, and allocators use intrusive pointers, `ArrayRef`, `SmallVector`, and exceptions |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depended-on-by | Dispatcher and boxed IValue code use type lists, bitsets, `flat_hash_map`, exceptions, and intrusive ownership |

## Runtime Behaviour

`ArrayRef` stores only a data pointer and length; callers pass it by value and keep ownership in the source container. Its checked accessors call `TORCH_CHECK`, so invalid `front`, `back`, `slice`, and `at` operations report c10 errors instead of raw undefined behaviour. `intrusive_ptr_target` keeps strong and weak counts in one atomic `uint64_t`, uses relaxed increments, acq-rel decrements, and reserves the high bit to record a preserved Python wrapper for tensor and storage objects. `Error` stores message text, context, and a `Backtrace`, then lazily formats `what()` and a backtrace-free string for C++ and Python exception conversion.

## Performance Profile

The utilities sit under every tensor operation, so they trade generality for predictable data layout. `ArrayRef` avoids allocation and copies while carrying shape, stride, and argument lists through APIs. `SmallVector` stores its first elements inline and grows out-of-line only after the inline capacity is exhausted, which matches the small ranks and short metadata lists used by tensors. `intrusive_ptr` places the reference count inside the object and removes the separate allocation and extra cache miss that a `std::shared_ptr` control block adds.

## Design Rationale

Chapter 02 describes tensor infrastructure as a flat metadata system, and `c10/util` provides the primitives that make that system practical. PyTorch uses non-owning views for API lists because shapes, strides, and dispatch argument lists are short-lived and already owned elsewhere. It uses intrusive ownership because tensors and storages dominate object traffic, and the framework controls their class layout. It centralizes errors and warnings in c10 so every layer, from allocator code to dispatcher code, reports consistent diagnostics without depending on Python.
