# `aten`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten` is the ATen tensor library root. It organizes the operator YAML declarations, the `TensorIterator` execution engine, the dispatch table bootstrap, and the kernel implementations under `aten/src/ATen/`. It is the C++ foundation of PyTorch below the Python bridge.

## Key Files

| File | Purpose |
|---|---|
| `aten/src/ATen/native/native_functions.yaml` | Declarative operator registry: every ATen operator signature, variants, and dispatch keys |
| `aten/src/ATen/TensorIterator.cpp` | `TensorIterator` (1736 lines): element-wise loop driver; handles broadcasting, dtype promotion, parallelism |
| `aten/src/ATen/Dispatch.h` | `AT_DISPATCH_ALL_TYPES` and related macros: dtype → kernel type dispatch |
| `aten/src/ATen/core/dispatch/Dispatcher.h` | `Dispatcher`: routes operator calls through dispatch key stack to backend kernels |
| `aten/CMakeLists.txt` | Build rules; drives `torchgen/gen.py` to produce generated headers under `build/aten/src/ATen/` |

## Public Interface

`at::Tensor` and its operator methods, `at::TensorIterator`, `at::Dispatcher::singleton()`, `at::dispatch_*` macros, and the operator registration macros (`TORCH_LIBRARY`, `TORCH_LIBRARY_IMPL`). See sub-ADRs for full detail.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | Sub-ADR covering operator declarations, TensorIterator, dispatch |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depended-on-by | Sub-ADR covering operator implementations |
| [aten/src/ATen/native/cpu](aten/src/ATen/native/cpu/ADR.md) | depended-on-by | Sub-ADR covering vectorized CPU kernels |
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl`, `DispatchKey`, `Storage`, `Allocator` |
| [torchgen](torchgen/ADR.md) | depends-on | `gen.py` generates headers consumed by ATen at build time |

## Runtime Behaviour

At process start the ATen dispatch table is populated by static initializers in generated `RegisterCPU.cpp` files (produced from `native_functions.yaml` by `torchgen`). Every `at::Tensor` operator call enters `Dispatcher::call()`, which looks up the registered kernel for the current dispatch key stack and calls it. `TensorIterator::build()` is called by most pointwise and reduction kernels to handle shape-broadcasting and output allocation before launching the inner loop. See sub-ADRs for detailed runtime behaviour of each sub-layer.

## Performance Profile

The dispatch lookup in `Dispatcher::call()` is a single indirect function call via a boxed or unboxed function pointer — low cost per op. `TensorIterator` amortizes loop setup across all elements; the primary hot path is the inner kernel loop in `aten/src/ATen/native/cpu/` (vectorized via AVX/AVX2/AVX-512 on CPU). Allocation costs come from `c10::Allocator` calls inside `TensorIterator::build()` for output tensors. See child ADRs for specifics.

## Design Rationale

`native_functions.yaml` as the single source of truth for the operator surface allows `torchgen` to generate dispatch registrations, Python bindings, and autograd glue consistently — editing the YAML propagates to all three layers. The ATen root is split into `src/ATen/` (declarations and engine), `src/ATen/native/` (implementations), and `src/ATen/native/cpu/` (vectorized paths) to separate the stable dispatch API from implementation-detail kernels that change frequently.
