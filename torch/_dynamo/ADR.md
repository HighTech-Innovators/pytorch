# `torch/_dynamo`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_dynamo` is TorchDynamo: a Python-level JIT that captures PyTorch programs into FX graphs by symbolically executing Python bytecode via CPython's PEP 523 frame evaluation hook. It is the front-end of `torch.compile`.

## Key Files

| File | Purpose |
|---|---|
| `torch/_dynamo/eval_frame.py` | Runtime entry point; `torch.compile()` wraps a function in an `OptimizedModule`; the C extension intercepts frames via PEP 523 |
| `torch/_dynamo/symbolic_convert.py` | Heart of Dynamo (6454 lines): `InstructionTranslator` symbolically executes bytecode instruction-by-instruction, maintains symbolic stack and `symbolic_locals` |
| `torch/_dynamo/output_graph.py` | `OutputGraph` owns the FX graph under construction, side-effects tracker, guards, shape environment; `compile_subgraph()` finalizes and calls the backend |
| `torch/_dynamo/guards.py` | Guard generation (5643 lines): runtime conditions (`TENSOR_MATCH`, `TYPE_MATCH`, `ID_MATCH`, etc.) that must hold for a cached compiled function to be reused |
| `torch/_dynamo/convert_frame.py` | `ConvertFrameAssert.__call__` checks caches, handles recompilation limits, calls `_compile()` → `trace_frame()` |
| `torch/_dynamo/bytecode_transformation.py` | Low-level bytecode manipulation: instruction patching, resume continuation generation |
| `torch/_dynamo/variables/` | `VariableTracker` subclass hierarchy: every Python value during tracing is wrapped here (`TensorVariable`, `ConstantVariable`, `NNModuleVariable`, etc.) |
| `torch/_dynamo/config.py` | All configuration flags; supports `config.patch()` as a decorator or context manager |
| `torch/_dynamo/trace_rules.py` | Per-function inline/skip/graph-break decisions |

## Public Interface

`torch.compile(fn, backend=..., mode=..., fullgraph=..., dynamic=...)`, `torch._dynamo.reset()`, `torch._dynamo.config.patch(...)`, `torch._dynamo.explain()`, `torch._dynamo.export()`, `torch._dynamo.testing.CompileCounter`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | `Graph`/`Node`/`GraphModule` IR produced by `OutputGraph` |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Default backend that receives compiled FX graphs |
| `torch._C._dynamo` | depends-on | C extension: PEP 523 frame hook, guard evaluation tree, cache entries |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Compiled autograd integration (`compiled_autograd.py`) |

## Runtime Behaviour

At `torch.compile(fn)` the function is wrapped; on the first call, the C extension (`eval_frame.c` via `_PyInterpreterState_SetEvalFrameFunc`) intercepts the frame and invokes `ConvertFrameAssert.__call__`. `InstructionTranslator` in `symbolic_convert.py` walks each bytecode instruction, wrapping encountered values in `VariableTracker` subclasses. Tensor operations become `call_function` nodes in the `OutputGraph` FX graph; non-traceable Python control flow triggers a graph break, which compiles the partial graph and generates a resume function via `resume_execution.py`. Guards are emitted alongside each compilation; on subsequent calls the C++ `RootGuardManager` evaluates them in a fast tree walk — on a cache hit the compiled code runs directly, bypassing Python entirely.

## Performance Profile

Guard evaluation on cache hit is the per-call overhead: the C++ guard tree (`guards.cpp`, ~7800 lines) runs dtype/shape/dispatch-key checks without crossing into Python. `FrameLocalsMapping` provides O(1) local variable access without dict materialization. Recompilation (cache miss) triggers full Python-level tracing through `symbolic_convert.py` — expensive and dominated by the 6454-line `InstructionTranslator` iteration. Guard explosion on dynamic-shape inputs forces repeated recompilation; `TORCH_LOGS="recompiles"` and `pgo.py` (profile-guided optimization) address this. Memory cost scales with the number of cached compilations stored in `ExtraState` per code object.

## Design Rationale

Using PEP 523 frame interception rather than AST rewriting means Dynamo works on arbitrary Python without requiring source access or decorator buy-in. Wrapping every value in a `VariableTracker` defers the inline/skip/graph-break decision per value rather than per function, enabling fine-grained partial graph capture. The Python-side guard specification (`guards.py`) is compiled into a C++ `GuardManager` tree at finalization — this separation keeps guard logic readable in Python while keeping the hot-path check entirely in C++.
