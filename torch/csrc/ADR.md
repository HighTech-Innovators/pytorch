# `torch/csrc`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc` is the Python↔C++ bridge — the C++ source of the `torch._C` extension module (`libtorch_python`). It registers Python types for tensors, dtypes, devices, storages, and streams; wires the autograd, JIT, and profiler subsystems into Python; and hosts the C++ frontend. It is the layer that makes thousands of ATen/c10 functions callable from Python.

## Key Files

| File | Purpose |
|---|---|
| `torch/csrc/Module.cpp` | `initModule()` (108KB) — orchestrates the entire `torch._C` surface initialization |
| `torch/csrc/autograd/python_variable.cpp` | `THPVariable`: the Python object wrapping `at::Tensor` |
| `torch/csrc/Dtype.cpp`, `Device.cpp`, `Layout.cpp` | Python type objects for `torch.dtype`, `torch.device`, `torch.layout` |
| `torch/csrc/Storage.cpp` | `torch.Storage` Python bindings |
| `torch/csrc/Exceptions.cpp` | Translates `c10::Error` into Python exceptions |
| `torch/csrc/autograd/` | C++ autograd engine and its Python bridge (see child ADR) |
| `torch/csrc/jit/` | TorchScript/JIT compiler C++ (see child ADR) |
| `torch/csrc/api/` | Python-free C++ frontend (see child ADR) |
| `torch/csrc/profiler/` | Kineto profiling integration exposed as `torch._C._profiler` |

## Public Interface

`initModule()`, `THPModule_initExtension()`, `THPVariable`, `THPVariable_Wrap()`, `THPVariable_Unpack()`, the `TorchMethods` `PyMethodDef` table, and the per-subsystem `init*` functions (`THPAutograd_initExtension`, JIT/distributed/profiler init entry points) invoked from `initModule()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Wraps `at::Tensor` and calls ATen operators |
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl`, `Device`, `ScalarType`, `Storage` |
| CPython + pybind11 | depends-on | Raw CPython API for `THPVariable`; pybind11 for newer subsystems |
| [torchgen](torchgen/ADR.md) | depends-on | Generates most tensor-method and `torch.*` Python bindings |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Python autograd calls into this bridge |

## Runtime Behaviour

At `import torch`, CPython loads `libtorch_python` and calls `initModule()` in `Module.cpp`, which runs subsystem initializers in a fixed order: core types (`THPVariable`, `THPFunction`, streams) first, then generated ATen bindings, then autograd hooks, JIT, storage/device, and (if enabled) distributed. `THPVariable_Wrap(at::Tensor)` allocates a Python object holding a `c10::MaybeOwned<at::Tensor>` in its `cdata` field; `THPVariable_Unpack()` extracts the tensor for C++ calls. Because initialization is ordered, a subsystem that needs `THPVariable` ready must appear after its registration — an `import torch` failure often points at module init rather than any operator.

## Performance Profile

Crossing the Python↔C++ boundary is the highest-cost boundary in the architecture: it requires GIL acquisition, `PythonArgs` argument parsing, `THPVariable_Unpack`/`THPVariable_Wrap` type conversion, and reference-count bookkeeping across both Python and C++ intrusive counts. This per-call overhead is why small-op-heavy Python loops are dominated by bridge cost rather than compute, and why `torch.compile` (which collapses many ops into one compiled call) helps. Native kernels release the GIL when they no longer touch Python objects, allowing parallel C++ execution; the autograd engine reacquires it only to run Python-defined hooks.

## Design Rationale

`THPVariable` is hand-written against the raw CPython API rather than pybind11 because it *is* the Python tensor object — its layout and lifecycle are part of the public API. Most operator bindings are generated (from schemas) because the surface is too large to maintain by hand and must stay synchronized with ATen. `Module.cpp` centralizes initialization order so cross-subsystem dependencies are satisfied deterministically; the trade-off is a 108KB monolith that every extension change must touch. The hybrid raw-CPython/pybind11 strategy trades a single consistent style for pragmatic per-subsystem choices.
