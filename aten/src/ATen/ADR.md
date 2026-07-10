# `aten/src/ATen`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen` ("A TENsor library") is PyTorch's C++ operator library. It owns the operator dispatcher (`core/dispatch/Dispatcher.h`), the element-wise/reduction execution engine (`TensorIterator`), the dtype-dispatch macros (`Dispatch.h`), the intra-op parallelism primitive (`Parallel.h`), and the public `at::Tensor` operator surface. It sits directly on top of `c10` and is consumed by the Python bridge.

## Key Files

| File | Purpose |
|---|---|
| `aten/src/ATen/core/dispatch/Dispatcher.h` | `Dispatcher::call`/`callBoxed`/`redispatch`; per-operator kernel lookup |
| `aten/src/ATen/core/dispatch/OperatorEntry.h` | Per-operator dispatch table keyed by `DispatchKey` |
| `aten/src/ATen/TensorIterator.h` / `.cpp` | Broadcasting, dtype promotion, dimension reordering, parallel loop scheduling |
| `aten/src/ATen/Dispatch.h` | `AT_DISPATCH_*` macros: runtime dtype → templated kernel switch |
| `aten/src/ATen/Parallel.h` | `at::parallel_for`, `get_num_threads`, `intraop_launch` |
| `aten/src/ATen/Context.h` / `.cpp` | Global runtime state (thread counts, backend flags, RNG) |
| `aten/src/ATen/Tensor.h` | The public `at::Tensor` handle |
| `aten/src/ATen/native/native_functions.yaml` | Declarative operator registry (signatures + dispatch intent) |
| `aten/src/ATen/ops/` | Generated per-operator declaration headers |

## Public Interface

`at::Tensor`, `c10::Dispatcher`, `OperatorHandle::call()` / `callBoxed()`, `TensorIterator`, `TensorIterator::borrowing_binary_op()`, `TensorIteratorBase::for_each()`, `AT_DISPATCH_ALL_TYPES_AND2`, `TORCH_META_FUNC`, `TORCH_IMPL_FUNC`, `at::parallel_for()`, `at::globalContext()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl`, `DispatchKeySet`, `ScalarType`, `Allocator` |
| [c10/util](c10/util/ADR.md) | depends-on | `ArrayRef`, `SmallVector`, `TORCH_CHECK`, `intrusive_ptr` |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Concrete kernel implementations registered into dispatch tables |
| [torchgen](torchgen/ADR.md) | depends-on | Generates `ops/`, registration code, and method bindings from YAML |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Python bindings call ATen operators |

## Runtime Behaviour

An operator call reaches an `OperatorHandle`; the typed path calls `Dispatcher::call(...)` and the boxed path calls `callBoxed()`, which computes a `DispatchKeySet` from the arguments via the operator's `DispatchKeyExtractor`, looks up the winning kernel with `entry.lookup(dispatchKeySet)`, and invokes it. Functionality layers (autograd, functionalize) peel keys off and call `Dispatcher::redispatch(...)` to reach the backend kernel. For element-wise ops the kernel builds a `TensorIterator`, whose `build()` calls `compute_strides()`, `reorder_dimensions()`, `allocate_or_resize_outputs()`, and `coalesce_dimensions()` before `for_each()` splits the work into `serial_for_each` (small) or `at::parallel_for` (large) ranges.

## Performance Profile

Dispatch is a bitmask computation plus a per-operator table lookup — cheap, but per-call, so it dominates at very small tensor sizes. `TensorIterator` setup (broadcast strides, dtype promotion via `AT_DISPATCH_*`, output allocation, dimension reordering) runs before any arithmetic, which is why tiny CPU ops can be setup-bound in profiles. `at::parallel_for` splits work by grain size; below the grain threshold it stays single-threaded to avoid scheduling overhead. Matrix ops delegate to MKL/OpenBLAS and convolutions to MKLDNN, so their cost lives in vendor libraries rather than in ATen loops. `allocate_or_resize_outputs()` is a per-op allocation site for freshly-created outputs.

## Design Rationale

`TensorIterator` centralizes broadcasting, type promotion, contiguity detection, parallelization, and vectorization so each element-wise kernel is a short lambda instead of ~200 lines of boilerplate; the trade-off is fixed microsecond-scale setup cost per call. The declarative `native_functions.yaml` registry keeps the operator surface generated and synchronized while leaving performance-sensitive bodies in ordinary C++. `AT_DISPATCH_*` macros solve runtime dtype selection at the cost of per-type binary size. Making `parallel_for` a shared primitive keeps CPU parallelism uniform rather than reinvented per operator.
