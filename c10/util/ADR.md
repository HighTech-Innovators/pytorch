# `c10/util`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/util` provides the C++ utility library that `c10/core` and ATen depend on: intrusive reference counting, array views, exception macros, numeric types (`BFloat16`, `Half`), thread-local state helpers, and diagnostic infrastructure. It has no dependencies on any other PyTorch component.

## Key Files

| File | Purpose |
|---|---|
| `intrusive_ptr.h` | `intrusive_ptr<T>` and `weak_intrusive_ptr<T>` — reference-counted smart pointer used for `TensorImpl`, `StorageImpl`, and all other core objects |
| `ArrayRef.h` | Non-owning span over a contiguous array; used pervasively for shape and stride arguments |
| `DimVector.h` | Small-buffer-optimised vector of `int64_t` dimensions; avoids heap allocation for rank ≤ 5 |
| `Exception.h` | `TORCH_CHECK`, `TORCH_INTERNAL_ASSERT`, `Error`, `TypeError`, `ValueError` macros and classes |
| `Backtrace.h` / `Backtrace.cpp` | Stack trace capture using platform `backtrace()` / libunwind |
| `BFloat16.h` / `BFloat16-inl.h` | `BFloat16` scalar type with arithmetic operators; no hardware intrinsics assumed |
| `typeid.h` | `TypeMeta` — runtime scalar-type metadata wrapper used by `TensorImpl` |
| `ThreadLocalDebugInfo.h` | Thread-local key-value debug info stack; used by autograd and distributed |
| `DeadlockDetection.h` / `DeadlockDetection.cpp` | Lock-order checking utilities for debug builds |
| `ApproximateClock.h` / `ApproximateClock.cpp` | Fast approximate wall-clock for profiler event timestamps |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::intrusive_ptr<T>` | Reference-counted pointer; `T` must inherit `intrusive_ptr_target`; refcount stored in `T` |
| `c10::weak_intrusive_ptr<T>` | Weak reference that does not prevent destruction |
| `c10::ArrayRef<T>` | Non-owning view; implicit from `std::vector<T>` and C arrays |
| `c10::SmallVector<T, N>` | `llvm::SmallVector` clone with inline storage |
| `c10::DimVector` | `SmallVector<int64_t, 5>` alias |
| `TORCH_CHECK(cond, ...)` | Throws `c10::Error` with formatted message if condition is false |
| `TORCH_INTERNAL_ASSERT(cond, ...)` | Unconditional assert; aborts in release builds |
| `c10::Error` | Base exception class; carries message plus optional backtrace |
| `c10::TypeMeta` | Runtime scalar-type metadata: size, alignment, copy/move constructors |
| `c10::ApproximateClock::getTime()` | Returns a fast timestamp without syscall overhead |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depended-on-by | c10/core includes ArrayRef, intrusive_ptr, Exception, DimVector from this directory |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen kernels use ArrayRef for shape arguments and intrusive_ptr for tensor lifetime |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | pybind11 bindings catch `c10::Error` and translate it to Python exceptions |

No external dependencies outside of standard C++ library and system headers.

## Runtime Behaviour

`intrusive_ptr` stores the reference count as a `std::atomic<uint32_t>` inside the target object (`intrusive_ptr_target`). Increment is `fetch_add(1, relaxed)`, decrement is `fetch_sub(1, acq_rel)` with a destroy call on zero. `weak_intrusive_ptr` uses a separate `weak_count` atomic field. `ArrayRef` is a struct of two raw pointers with no heap allocation; it is passed by value and is zero-cost to construct from a `std::vector`. `TORCH_CHECK` calls into `c10::Error`'s constructor, which calls `Backtrace::get()` only when `FLAGS_torch_show_cpp_stacktraces` is set, making the normal error path cheap.

## Performance Profile

- **Allocation sites**: `intrusive_ptr` itself performs no heap allocation — the refcount lives inside the target object. The cost is an `atomic<uint32_t>` increment per tensor copy, which is a cache-line bounce when the `TensorImpl` is shared across threads.
- **Synchronization costs**: The `acq_rel` fence on decrement is the primary synchronization cost; it becomes a bottleneck when many threads simultaneously release references to the same tensor.
- **Data movement**: None — this is a pure utility library with no data-moving operations.
- **Redundant or repeated work**: `ApproximateClock::getTime()` calls `clock_gettime(CLOCK_MONOTONIC_COARSE)` on Linux, which is a vDSO call — fast but not zero-cost. It is called once per profiler event, so frequency tracks profiler verbosity.

## Design Rationale

`intrusive_ptr` is preferred over `std::shared_ptr` because it places the refcount inside the managed object, reducing the allocation count from two (object + control block) to one and improving cache locality for types like `TensorImpl` that are always heap-allocated. `ArrayRef` is the standard "span" before `std::span` was available and is retained for compatibility. `DimVector`'s inline size of 5 matches the median tensor rank in typical vision workloads, avoiding heap allocation in the common case.
