# `tools/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools/autograd` is a build-time code-generation subsystem. It reads `derivatives.yaml` (gradient formulas for every differentiable ATen operator) and emits the C++ autograd dispatch glue consumed by `torch/csrc/autograd/`.

## Key Files

| File | Purpose |
|---|---|
| `tools/autograd/derivatives.yaml` | Declares gradient formulas for every differentiable ATen function (3288 lines); maps operator name → input gradient expressions |
| `tools/autograd/gen_autograd.py` | Top-level codegen driver: reads YAML, calls subgenerators, writes output files |
| `tools/autograd/gen_variable_type.py` | Generates `VariableType.h/cpp`: the dispatch layer that wraps ATen ops to record autograd graph edges |
| `tools/autograd/gen_autograd_functions.py` | Generates `generated/autograd/Functions.h/cpp`: backward `Node` subclasses for each op |
| `tools/autograd/gen_inplace_or_view_type.py` | Generates the `InplaceOrView` dispatch key handlers |
| `tools/autograd/load_derivatives.py` | Parses and validates `derivatives.yaml` into structured Python objects |
| `tools/autograd/templates/` | Jinja2/C++ templates used by the generators |

## Public Interface

`tools/autograd` has no runtime Python interface. Its output artifacts — `build/aten/src/ATen/generated/VariableType.cpp`, `build/aten/src/ATen/generated/autograd/Functions.h/cpp` — are consumed at C++ compile time. `gen_autograd.py` is invoked by `torchgen/gen.py` as part of the build.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | `torchgen/model.py` types and YAML loading used in `load_derivatives.py`; `gen_autograd.py` is called from `torchgen/gen.py` |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depended-on-by | Generated `VariableType.cpp` and `Functions.h/cpp` are compiled into `libtorch_python` |
| Build system (CMake/Ninja) | depended-on-by | Build rules invoke `gen_autograd.py` before C++ compilation |

## Runtime Behaviour

`tools/autograd` runs only at build time. `gen_autograd.py` invokes `load_derivatives.py` to parse `derivatives.yaml`, cross-references the operator list from `torchgen`, then runs `gen_variable_type.py` and `gen_autograd_functions.py` to write C++ source files into `build/`. These generated files register autograd dispatch keys and define backward `Node` subclasses; they are compiled into `libtorch_python` and do not exist at Python import time. No `tools/autograd` module is imported at runtime.

## Performance Profile

Build-time cost only: YAML parsing and template rendering in `gen_autograd.py` scales with the number of operators in `derivatives.yaml` (~3288 lines covering hundreds of operators). This is dominated by CMake's dependency tracking and C++ compilation of the generated files, not by the Python codegen itself. At runtime the generated C++ is as efficient as hand-written dispatch glue; there is no interpreted path.

## Design Rationale

Expressing gradient formulas in a declarative YAML file rather than hand-writing C++ for each operator keeps the ~hundreds of gradient definitions maintainable and auditable by non-C++ contributors. The codegen approach generates both the recording glue (`VariableType.cpp`) and the backward `Node` classes (`Functions.h/cpp`) from the same YAML source, ensuring they stay consistent. Delegating the heavy structural types and YAML loading to `torchgen/model.py` avoids duplicating operator schema parsing logic between the two code-generation subsystems.
