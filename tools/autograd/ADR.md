# `tools/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools/autograd` generates PyTorch's differentiable operator layer from `native_functions.yaml` and `tools/autograd/derivatives.yaml`. It emits Autograd dispatch-key wrappers, `torch::autograd::Node` subclasses, generated `Functions.h/cpp`, `VariableType` shards, ADInplaceOrView wrappers, view replay helpers, tracing wrappers, variable factories, and Python bindings. Book chapter 05 describes forward graph construction through generated `VariableType` wrappers and backward execution through `Node::apply()`; this directory writes those generated classes and wrapper bodies.

## Key Files

| File | Purpose |
|---|---|
| `gen_autograd.py` | Top-level autograd generator that loads derivative formulas, matches them to native functions, and delegates to all autograd subgenerators |
| `derivatives.yaml` | Declarative gradient and forward-mode derivative formulas keyed by ATen schemas and optional Autograd dispatch keys |
| `load_derivatives.py` | Parses `derivatives.yaml`, validates formulas against native schemas, computes saved inputs/outputs, and creates `DifferentiabilityInfo` objects |
| `gen_variable_type.py` | Generates Autograd dispatch wrappers that save inputs, redispatch to lower keys, attach `grad_fn`, and register kernels under Autograd keys |
| `gen_autograd_functions.py` | Generates `torch::autograd::Node` subclasses with saved variables, `apply()`, compiled-autograd hooks, and release logic |
| `gen_inplace_or_view_type.py` | Generates ADInplaceOrView wrappers, view metadata handling, view replay functions, and in-place/view registrations |
| `gen_python_functions.py` | Generates Python bindings and return-type structures for generated variable functions |
| `templates/VariableType.cpp` | Template for sharded generated `VariableType_*.cpp` files and wrapper registration blocks |

## Public Interface

The build calls `python -m tools.autograd.gen_autograd <native_functions.yaml> <tags.yaml> <out> <autograd_dir>`. `gen_autograd()` emits C++ autograd files, and `gen_autograd_python()` emits Python-facing generated functions. Subgenerators expose `gen_variable_type()`, `gen_inplace_or_view_type()`, `gen_autograd_functions_lib()`, `gen_autograd_functions_python()`, `gen_trace_type()`, `gen_variable_factories()`, and `gen_view_funcs()`. `load_derivatives()` returns a map from `FunctionSchema` to per-dispatch-key `DifferentiabilityInfo` plus the set of Autograd keys that require `TORCH_LIBRARY_IMPL` registrations.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Parses native schemas, groups view/copy variants, filters selected operators, and provides file emission utilities |
| [torchgen/api](torchgen/api/ADR.md) | depends-on | Uses C++ signatures, autograd differentiability models, saved attributes, type bindings, and schema-to-expression translation |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | generates-for | Emits generated `VariableType`, `Functions`, view, factory, and Python binding files consumed by the autograd runtime from book chapter 05 |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on/generates-for | Registers Autograd dispatch-key kernels and redispatches through generated `at::_ops` wrappers into lower-priority kernels |
| [torch/autograd](torch/autograd/ADR.md) | generates-for | Provides Python-visible autograd function bindings and generated variable methods backing public autograd behavior |

## Runtime Behaviour

At generation time, `gen_autograd.py` loads derivative definitions, parses native functions, filters selected training operators, matches each `NativeFunction` with differentiability information, and writes generated files through `FileManager`. `load_derivatives.py` replaces formula references to inputs and outputs with saved attributes, records which named gradients are used, adds generated derivatives for `{view}_copy` variants, and caches deterministic parse results. Generated `VariableType` wrappers implement the book chapter 05 forward path: they check derivative requirements, save needed tensors or scalar metadata, redispatch to the underlying kernel with Autograd removed, create a generated `Node`, and attach it to differentiable outputs.

## Performance Profile

Autograd generation pays build-time cost proportional to the number of native schemas and derivative entries, then shards `VariableType.cpp` into ten generated files because the template records that the old monolithic file exceeded 36,000 lines and slowed incremental rebuilds. Generated wrappers avoid recording backward nodes for operators in `DONT_REQUIRE_DERIVATIVE`, for operators without differentiable outputs, and for unsupported missing derivatives that register an AutogradNotImplemented boxed fallback. At runtime, the generated code saves only formula-referenced inputs/outputs and uses `grad_input_mask` so backward formulas compute requested input gradients only. The build-time generator therefore spends extra analysis to reduce graph memory and backward work in the book chapter 05 execution model.

## Design Rationale

PyTorch expresses derivatives declaratively because the same operator schema must drive reverse-mode nodes, forward-mode formulas, Python bindings, view handling, tracing, and generated dispatch registrations. `derivatives.yaml` keeps mathematical formulas beside schema names, while `load_derivatives.py` turns them into typed saved-variable plans that C++ generation can validate. ADInplaceOrView generation lives separately because aliasing and in-place metadata rules differ from ordinary differentiable operators and must stay aligned with `torch/csrc/autograd/autograd_not_implemented_fallback.cpp`. The design matches book chapter 05: forward execution remains eager and dynamic, while generated code supplies the repeatable boilerplate for graph construction and backward `Node::apply()` implementations.
