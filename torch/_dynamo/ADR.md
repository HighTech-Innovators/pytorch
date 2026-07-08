# `torch/_dynamo`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_dynamo` is the Python compiler frontend behind `torch.compile()`, as mapped in book chapter 07. It installs a CPython PEP 523 frame-evaluation hook, selects tensor-containing frames, symbolically executes Python bytecode with `InstructionTranslator`, records tensor operations into an FX graph through `OutputGraph`, emits replacement bytecode through `codegen.py`, and protects cached compiled code with runtime guards. It owns graph-break handling, resume-function generation, source provenance, side-effect tracking, guard construction, backend registration, and the user-facing optimization decorators exported from `__init__.py`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public Dynamo namespace; exports `optimize`, `optimize_assert`, `export`, `explain`, backend registration, graph-break controls, dynamic-shape markers, and reset helpers |
| `eval_frame.py` | Runtime entry point for optimization contexts; wraps functions/modules and calls `torch._C._dynamo.eval_frame.set_eval_frame` |
| `convert_frame.py` | Per-frame conversion driver; filters frames, manages cache/recompile limits, and invokes bytecode tracing |
| `symbolic_convert.py` | Bytecode symbolic interpreter; `InstructionTranslator` maintains symbolic stack, locals, globals, speculation, and opcode dispatch |
| `output_graph.py` | Owns the FX graph, `SubgraphTracer`, shape environment, graph arguments, guards, side effects, and backend compilation boundary |
| `guards.py` | Builds guard code and C++ guard-manager trees for type, identity, tensor metadata, dispatch-key, functorch, and global-state checks |
| `codegen.py` | Emits replacement Python bytecode that loads graph inputs, calls compiled graphs, unpacks outputs, and replays side effects |
| `resume_execution.py` | Synthesizes continuation code objects that resume Python execution after graph breaks |
| `source.py` | Defines value provenance objects used to reconstruct runtime values and install guards |
| `variables/` | `VariableTracker` implementations for tensors, constants, modules, lists, dicts, user functions, builtins, and user objects |

## Public Interface

`torch._dynamo.optimize()` and `torch._dynamo.optimize_assert()` wrap callables with a backend compiler. `torch._dynamo.export()` captures an FX graph for export flows, while `explain()` reports graph breaks and guards. Decorators such as `allow_in_graph`, `disallow_in_graph`, `disable`, `graph_break`, `mark_dynamic`, `mark_static`, and `substitute_in_graph` control tracing behavior. Backend utilities `register_backend`, `lookup_backend`, and `list_backends` connect Dynamo to Inductor and other compilers. `reset()` and `reset_code_caches()` clear in-memory frame, guard, backend, and resume caches for tests and diagnostics.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | `OutputGraph` builds `GraphModule` objects and records operations as FX nodes |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | The default `torch.compile` backend receives Dynamo-produced FX graphs |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Guards track functorch dynamic-layer state, and AOTAutograd consumes Dynamo graphs in compile flows |
| [torch/csrc/dynamo](torch/csrc/dynamo/ADR.md) | depends-on | C++ frame evaluation, cache entries, shadow-frame execution, and guard managers implement the hot runtime |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Tracing guards grad mode, saved-tensor hooks, forward-ad levels, and compiled autograd state |
| [c10/core](c10/core/ADR.md) | depended-on-by | Tensor guards check dtype, device, layout, dispatch key set, sizes, strides, and aliasing metadata from tensor internals |

## Runtime Behaviour

A compiled call enters `eval_frame.py`, which installs or updates the C eval-frame callback and associates an optimization context with the Python code object. The C runtime calls the Python conversion callback on a cache miss; `ConvertFrameAssert.__call__` rejects generated code, generator frames, non-tensor frames, disabled frames, and cache-limit cases before allocating a compile id and tracing the frame. `InstructionTranslator` walks bytecode instructions, wraps Python values in `VariableTracker` subclasses, records tensor operations through proxies into `OutputGraph`, and raises structured graph-break exceptions for unsupported operations. At a graph break, Dynamo compiles the partial FX graph, emits bytecode that calls the compiled graph, and uses `resume_execution.py` to build a continuation function for the rest of the original frame.

On subsequent calls, the C cache evaluates a guard tree before Python tracing runs. `guards.py` installs checks for object identity, type, sequence length, tensor dtype/device/size/stride, dispatch keys, global grad/autocast state, functorch stack state, and subclass metadata; a guard hit executes the cached compiled bytecode through a shadow frame, while a guard miss records the failure reason and retraces under the recompile policy.

## Performance Profile

The runtime hot path sits in `torch/csrc/dynamo`: cache lookup, C++ guard evaluation, frame-local access, and compiled-code shadow-frame execution avoid Python dict materialization and pybind overhead. Python files in this directory run mostly at compile time, but `eval_frame.py` marks frame-evaluation functions as performance-critical because they execute on every intercepted frame. Compile-time cost concentrates in `symbolic_convert.py`, `output_graph.py`, fake-value creation, guard construction, and backend invocation; `BytecodeTracingTimings` records nanosecond totals for fake values, proxy creation, wrapping, and variable-builder calls. Cache quality dominates steady-state speed: stable guards amortize tracing and backend compilation, while dynamic shapes or changing Python objects trigger recompilation until cache limits force eager fallback.

## Design Rationale

Dynamo traces CPython bytecode instead of requiring a new frontend language because PyTorch programs use normal Python modules, closures, containers, control flow, and side effects. FX is the output format because it preserves Python-level operator targets and source locations while remaining easy for AOTAutograd and Inductor to transform. Guards are explicit runtime contracts because compiled code specializes on Python object identity, tensor metadata, global modes, and symbolic-shape constraints; invalid assumptions cause retracing instead of silent wrong results. Graph breaks preserve Python semantics: unsupported code executes eagerly, and resume bytecode lets compilation continue after the eager region without forcing users to rewrite models.
