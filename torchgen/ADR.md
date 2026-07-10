# `torchgen`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen` is the YAML-driven code-generation engine for PyTorch operators. It reads `aten/src/ATen/native/native_functions.yaml` and emits C++ dispatch registrations, Python bindings, autograd glue, and structured kernel headers consumed by `aten/`, `torch/csrc/`, and `tools/autograd/`.

## Key Files

| File | Purpose |
|---|---|
| `torchgen/gen.py` | Top-level driver (116KB): reads YAML, constructs `NativeFunction` objects, calls all destination generators, writes output files |
| `torchgen/model.py` | Domain model (124KB): `NativeFunction`, `FunctionSchema`, `DispatchKey`, `BackendIndex`, `Argument`, `Type` — the typed Python representation of the YAML schema |
| `torchgen/api/` | API translation: `dispatcher.py`, `cpp.py`, `native.py`, `structured.py` — converts `NativeFunction` to C++ signatures for each calling convention |
| `torchgen/dest/` | Output generators: `register_dispatch_key.py` (dispatch registrations), `native_functions.py` (declarations), etc. |
| `torchgen/decompositions/` | Decomposition registrations emitted for composite ops |
| `torchgen/context.py` | `with_native_function` / `native_function_manager` context helpers for error-reporting during codegen |

## Public Interface

`torchgen` is a build-time tool; it has no runtime Python API. `gen.py` is invoked by `aten/CMakeLists.txt` and `tools/autograd/gen_autograd.py` during the build. `torchgen/model.py` types are imported by `tools/autograd/load_derivatives.py` at build time.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | Generated headers (`RegisterCPU.cpp`, operator declarations) are compiled into ATen |
| [tools/autograd](tools/autograd/ADR.md) | depended-on-by | `load_derivatives.py` imports `torchgen/model.py` types; `gen_autograd.py` is called from `gen.py` |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Generated Python binding files are compiled into `libtorch_python` |
| Build system (CMake) | depended-on-by | `aten/CMakeLists.txt` invokes `gen.py` as a build step |

## Runtime Behaviour

`torchgen` runs only at build time. `gen.py` reads `native_functions.yaml` with `yaml.safe_load`, constructs typed `NativeFunction` objects via `model.py`, groups them by `BackendIndex`, and drives each destination generator (`dest/register_dispatch_key.py`, `dest/native_functions.py`, etc.) to write C++ files into `build/aten/src/ATen/`. The generated files include `RegisterCPU.cpp` (static initializers that populate the dispatch table), `NativeFunctions.h` (operator declarations), and Python-binding source files. No `torchgen` module is imported at Python runtime.

## Performance Profile

Build-time cost only: parsing `native_functions.yaml` and rendering hundreds of C++ files scales linearly with operator count. The generated C++ has the same performance characteristics as hand-written dispatch code — `torchgen` emits optimal boilerplate, not interpreted wrappers. CMake incremental builds skip regeneration when `native_functions.yaml` is unchanged, so normal development iteration does not pay the full codegen cost.

## Design Rationale

Centralizing all operator metadata in a single YAML file and driving all generated outputs from one tool ensures that operator signatures, dispatch registrations, and Python bindings stay in sync. Representing the schema in `model.py` as typed Python dataclasses (rather than raw dicts) catches schema errors at codegen time rather than at C++ compile time. The `api/` layer translates one canonical `NativeFunction` into multiple C++ calling-convention signatures (`dispatcher`, `cpp`, `native`, `structured`) — each calling convention has its own translation module rather than a single complex switch.
