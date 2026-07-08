# `torch/nativert`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nativert` owns the NativeRT runtime for exported models. It provides the C++ `ModelRunner`, Python bindings for invoking it, and backend-delegation shims for lowered exported programs.

## Key Files

| File | Purpose |
|---|---|
| `ModelRunner.h` | Declares `ModelRunner`, its `run` APIs, `numOutputs`, and weight-loading hook points |
| `ModelRunner.cpp` | Loads PT2 archives, deserializes `ExportedProgram`, builds the runtime graph, loads weights, and drives `Executor` inference |
| `python/Bindings.cpp` | Exposes `PyModelRunner` to Python and converts `py::args` and `py::kwargs` into `c10::IValue` inputs |
| `backends/_lowered_aoti_module.py` | Defines `LoweredBackendModule`, a Python delegate wrapper whose `forward` calls `executorch_call_delegate` |
| `__init__.py` | Marks the package root for the runtime namespace |

## Public Interface

The public symbols are `ModelRunner`, `run`, `runWithFlatInputsAndOutputs`, `numOutputs`, the Python-bound `PyModelRunner`, and `LoweredBackendModule.forward`. `LoweredBackendModule.backend_id`, `module_name`, and `original_module` expose delegation metadata alongside the callable interface.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/export](torch/export/ADR.md) | depends-on | `ModelRunner.cpp` parses a serialized `torch::_export::ExportedProgram` and reads archive constants from `pt2_archive_constants.h` |
| [torch/compiler](torch/compiler/ADR.md) | depended-on-by | the runtime executes delegated compiled graphs, including AOTInductor-lowered modules wrapped by `LoweredBackendModule` |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Python bindings convert Python objects with `torch::jit::toIValue` and return values with `torch::jit::createPyObjectForStack` |

## Runtime Behaviour

`ModelRunner::ModelRunner` first calls `register_kernel_handlers()`, opens the PT2 archive through `caffe2::serialize::PyTorchStreamReader`, parses the model JSON into `torch::_export::ExportedProgram`, and records payload paths from the weights and constants configs. It then converts the serialized graph with `jsonToGraph`, derives `inputSpec_` and `outputSpec_` with `itreeSpecLoads`, applies default device placement, runs `selectScalarOverload(graph_.get())`, loads immutable weights, and constructs an `Executor`. `ModelRunner::run` enters `c10::InferenceMode`, executes the graph through `executor_->execute(args, kwargs, inputSpec_)`, and reconstructs the structured result with `itreeUnflatten`. `Bindings.cpp` mirrors the same call path for Python by translating `py::args` and `py::kwargs` into `c10::IValue` containers and wrapping the returned stack back into Python objects.

## Performance Profile

Archive parsing and graph construction happen once in the constructor, so steady-state inference avoids repeating JSON parsing, weight discovery, and graph-pass setup. `runWithFlatInputsAndOutputs` is the lower-overhead call path because it skips pytree unflattening and hands an owned flat input vector directly to the `Executor`. The overview document describes pooled execution frames and immutable shared `Weights`, which let concurrent inference reuse preallocated runtime state instead of rebuilding per-call structures. Delegated execution through `LoweredBackendModule.forward` moves expensive kernel work into compiled backends while keeping the Python call boundary minimal.

## Design Rationale

NativeRT is built around an explicit runtime graph and executor instead of reusing eager-mode Python execution. That design lets exported programs, delegated subgraphs, and shared immutable weights run under one inference-oriented runtime with predictable threading and low Python overhead.
