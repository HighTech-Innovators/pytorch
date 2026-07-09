# `aten/src/ATen`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen` (A TENsor library) implements all PyTorch tensor operations: the multi-backend operator dispatcher, native C++ kernel implementations, and the code-generated operator registration glue. It is the performance-critical layer between the Python API and the hardware.

## Key Files

| File | Purpose |
|---|---|
| `core/dispatch/Dispatcher.h` | Singleton dispatcher: routes every operator call through `DispatchKeySet` lookup to the registered kernel; `OperatorHandle`, `TypedOperatorHandle<FuncType>`, `callBoxed` |
| `core/dispatch/OperatorEntry.h` | Per-operator registration table: one `KernelFunction` slot per `DispatchKey`; stores registered kernels and fallbacks |
| `native/native_functions.yaml` | 16,230-line YAML schema defining ~2,585 operator signatures, dispatch table entries, structured-kernel flags, and per-backend enablement |
| `Context.h` / `Context.cpp` | Global ATen context (`at::globalContext()`): cuBLAS/cuDNN handle management, default dtype, deterministic-mode flag |
| `TensorIterator.h` | `TensorIteratorBase` — broadcast-aware loop driver; all pointwise kernels use it to walk over operands with optimal loop ordering and dtype casting |
| `Dispatch.h` | `AT_DISPATCH_ALL_TYPES` family of macros: switch-to-template dispatch over scalar types; used in every native kernel |
| `CPUGeneratorImpl.h` / `CPUGeneratorImpl.cpp` | CPU random number generator state; seeded from `std::random_device`; used by `at::randn`, `at::rand`, `at::randint` |
| `native/BinaryOps.cpp` | Representative native kernel file: implements `add`, `sub`, `mul`, `div` using `TensorIterator` |
| `native/LinearAlgebra.cpp` | BLAS-backed operations: `mm`, `bmm`, `addmm`; dispatches to BLAS via `at::blas::gemm` |
| `DLConvertor.h` / `DLConvertor.cpp` | Conversion to/from DLPack `DLTensor` format for cross-framework interop |
| `record_function.h` | `RecordFunction` — profiler hook called at operator entry/exit; `RECORD_FUNCTION(name, args)` macro used throughout |

## Public Interface

| Symbol | Description |
|---|---|
| `at::Tensor` | User-facing tensor class wrapping `c10::TensorImpl` via `intrusive_ptr`; all arithmetic operators delegate to ATen dispatch |
| `c10::Dispatcher::singleton()` | Global dispatcher instance; `call<FuncType>(op, args...)` routes to the best-matching kernel |
| `c10::OperatorHandle` | Opaque handle to a registered operator; obtained via `Dispatcher::findSchemaOrThrow` |
| `at::TensorIteratorBase` | Loop driver for pointwise kernels; `build()`, `for_each()`, `serial_for_each()` |
| `at::globalContext()` | Returns the singleton `Context`; grants access to backend handles and global flags |
| `at::native::*` | Namespace containing all native kernel implementations (e.g., `at::native::add`, `at::native::mm`) |
| `RECORD_FUNCTION(name, inputs)` | Fires a `RecordFunction` event; consumed by the PyTorch profiler |
| `at::fromDLPack(dlTensor)` | Creates an `at::Tensor` from a DLPack capsule without copying data |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl`, `Storage`, `Allocator`, `DispatchKey`, `DispatchKeySet`, `Device`, `ScalarType` |
| [c10/util](c10/util/ADR.md) | depends-on | `ArrayRef`, `intrusive_ptr`, `Exception`, `DimVector` |
| [torchgen](torchgen/ADR.md) | depended-on-by | torchgen reads `native_functions.yaml` and generates `RegisterDispatchKey.cpp`, structured kernel stubs, and Python bindings |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | pybind11 bindings wrap `at::Tensor` and call through the dispatcher |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Autograd kernels are registered against `Autograd` dispatch keys; the C++ engine calls `at::Tensor::backward()` |
| BLAS/LAPACK libraries | depends-on | `native/LinearAlgebra.cpp` calls `cblas_sgemm`, `LAPACKE_ssyev` etc. |
| CUDA runtime / cuBLAS / cuDNN | depends-on | CUDA native kernels in `native/cuda/` call CUDA APIs; `Context.cpp` manages cuBLAS handles |

## Runtime Behaviour

Every `at::Tensor` operation calls `Dispatcher::singleton().call<FuncType>(op, args...)`. The dispatcher extracts the `DispatchKeySet` from the tensor arguments using `dispatchKeyExtractor`, finds the highest-priority key, and looks up the registered `KernelFunction` in `OperatorEntry`'s dispatch table. This lookup is a 64-bit `tzcnt` + array index — O(1). Fallback kernels handle cross-cutting concerns: `CompositeImplicitAutograd` kernels decompose to simpler operations, `Autograd` kernels wrap calls with gradient-tape recording, and `Python` dispatch allows Python-level overrides. `TensorIterator::build()` computes broadcast output shapes and optimal stride orderings before the loop runs; for contiguous same-device tensors this is a near-zero-cost fast path.

## Performance Profile

- **Allocation sites**: every non-in-place tensor operation allocates a new `TensorImpl` and calls `GetAllocator(device)->allocate(n_bytes)`. For CPU tensors this goes through `CPUAllocator`; for CUDA through `CUDACachingAllocator`. These are the dominant allocation hot spots in training.
- **Synchronization costs**: The dispatcher's `OperatorEntry` is protected by a `c10::LeftRight<>` lock-free reader structure that allows concurrent reads without locking. Writes (kernel registration at startup) take a full lock. `globalContext()` acquires a `std::mutex` when creating or accessing cuBLAS/cuDNN handles.
- **Data movement**: `native/Copy.cpp` implements `at::copy_` and dispatches H2D/D2H transfers through `CUDAStream::getCurrentCUDAStream()`. `DLConvertor` performs zero-copy interop by sharing the data pointer.
- **Redundant or repeated work**: `TensorIterator::build()` recomputes broadcast shapes on every call; for common fixed-shape models this is a non-trivial overhead. `RECORD_FUNCTION` fires on every operator call; when the profiler is inactive this reduces to a branch-predicted-false early exit.

## Design Rationale

The operator schema is defined in `native_functions.yaml` rather than in C++ headers so that `torchgen` can generate consistent registration code for every backend, Python binding, and mobile build variant from a single source of truth. `TensorIterator` centralises broadcast logic so that the ~2,500 native kernels do not each implement their own shape-checking and striding loops. The `CompositeImplicitAutograd` fallback allows complex operators to be defined as compositions of simpler ones while still getting autograd support automatically, at the cost of potentially less efficient gradient computation.
