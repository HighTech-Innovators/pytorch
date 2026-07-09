# `torch/_dynamo`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_dynamo` is TorchDynamo: a Python-level JIT compiler that hooks into CPython's frame evaluation API (PEP 523) to intercept bytecode execution, extract contiguous PyTorch operation sequences as FX graphs, and dispatch them to a configurable compilation backend (default: TorchInductor). It is the entry point for `torch.compile()`.

## Key Files

| File | Purpose |
|---|---|
| `eval_frame.py` | Core frame evaluation handler: registers the custom `eval_frame_callback` via `torch._C._dynamo.eval_frame`; `optimize`, `disable`, `run`, `export` decorators; `OptimizedModule`; guard cache management |
| `convert_frame.py` | `ConvertFrame` and `ConvertFrameAssert` — transforms a Python frame into an FX graph; `Tracker` tracks code objects; error handling and fallback |
| `bytecode_transformation.py` | Bytecode rewriting: `transform_code_object` modifies `co_code` to redirect execution through Dynamo's tracer |
| `bytecode_analysis.py` | Static analysis of CPython bytecode: liveness, control flow, variable tracking |
| `symbolic_convert.py` | `InstructionTranslator` — symbolic Python interpreter that traces through bytecode and emits FX graph nodes; handles `graph_break` when Python features are unsupported |
| `guards.py` | Guard specification and compilation: `GuardBuilder`, `GuardedCode`; guards are Python predicates that validate whether a cached compiled graph is still valid |
| `cache_size.py` | Guard cache eviction policy: limits recompilation count per code object |
| `config.py` | ~100 configuration flags controlling Dynamo behaviour: `assume_static_by_default`, `dynamic_shapes`, `cache_size_limit`, `suppress_errors` |
| `backends/` | Registered compilation backends: `inductor` (default), `eager`, `aot_eager`, `nvfuser`, etc. |
| `compiled_autograd.py` | Compiled backward: traces the autograd engine execution itself through Dynamo |
| `decorators.py` | `allow_in_graph`, `disallow_in_graph`, `mark_dynamic`, `mark_static`, `graph_break` decorators |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.compile(fn, backend, mode, dynamic)` | Entry point; wraps `fn` with `OptimizedModule` or `_OptimizedModule`; calls `eval_frame.optimize` |
| `torch._dynamo.optimize(backend)` | Lower-level decorator; installs the frame eval hook for one function |
| `torch._dynamo.disable(fn)` | Prevents Dynamo from compiling `fn`; passthrough to Python |
| `torch._dynamo.export(fn, *args)` | Runs Dynamo tracing and returns the FX `GraphModule` rather than executing it |
| `torch._dynamo.mark_dynamic(tensor, dim)` | Tags a tensor dimension as dynamic (unbounded shape) |
| `torch._dynamo.mark_static(tensor)` | Tags all tensor dimensions as static (specialised shapes) |
| `torch._dynamo.graph_break()` | Forces a graph break at the call site |
| `torch._dynamo.reset()` | Clears all guard caches and resets recompilation counters |
| `torch._dynamo.explain(fn, *args)` | Returns a human-readable summary of graph breaks and guard structure |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._dynamo.eval_frame` | depends-on | C-extension that installs the PEP 523 frame evaluation callback; `set_eval_frame`, `reset_code` |
| [torch/fx](torch/fx/ADR.md) | depends-on | Output IR is a `torch.fx.Graph`; Dynamo uses `GraphModule` and `Proxy` internally |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | AOT autograd (`aot_autograd`) is called after Dynamo extracts the graph; functorch's `make_fx` is used to trace the forward+backward |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Dynamo dispatches the extracted `GraphModule` to Inductor's `compile_fx` |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Guard evaluation uses `torch._C._dynamo.guards` C-extension; `PyInterpreterHooks` for reentrant tracing |

## Runtime Behaviour

When `torch.compile(fn)` is called, Dynamo installs a custom frame evaluation function via `torch._C._dynamo.eval_frame.set_eval_frame`. On the first call to `fn`, CPython calls Dynamo's callback instead of the default `_PyEval_EvalFrameDefault`. `ConvertFrame.call` processes the frame: `symbolic_convert.InstructionTranslator` symbolically executes the bytecode instruction-by-instruction, tracking tensor arguments as `Proxy` nodes and Python scalars as constants. When an unsupported operation (graph break) is encountered, Dynamo stops the current graph, compiles what was captured, and resumes Python execution normally. Compiled graphs and their associated guards are stored in a per-code-object cache. On subsequent calls, Dynamo checks the guards (Python predicates on tensor shapes/dtypes/device); if all pass, the cached compiled function is called directly; if any fail, recompilation occurs.

## Performance Profile

- **Allocation sites**: guard checking allocates no objects on cache hit — it evaluates pre-compiled guard functions. Each recompilation allocates new `GuardedCode` objects and (via Inductor) new compiled C++/Triton functions.
- **Synchronization costs**: Dynamo is single-threaded per Python interpreter; the frame evaluation hook is installed per-thread. The guard cache uses a `dict` keyed by `code` object identity — O(1) lookup.
- **Data movement**: symbolic tracing with `Proxy` objects produces no real tensor data movement. Compiled graph execution passes real tensors through the compiled function with standard ATen dispatch.
- **Redundant or repeated work**: recompilation triggered by shape changes is the primary recurring cost. `assume_static_by_default=True` (default) specialises on the first input shape; dynamic inputs require `mark_dynamic` to avoid repeated recompilation.

## Design Rationale

PEP 523 frame hooking rather than AST rewriting was chosen because AST-level transforms cannot capture Python semantics at runtime (e.g., closures, dynamic attribute access, `__torch_function__` overrides). Bytecode-level interception sees exactly what CPython sees. Graph breaks — rather than failing on unsupported operations — allow Dynamo to compile the portions it can and fall back to Python for the rest, enabling progressive adoption. The guard system enables cache reuse across calls with compatible shapes/dtypes without requiring a global compilation model.
