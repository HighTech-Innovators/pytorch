# `torch/fx`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/fx` is PyTorch's graph intermediate representation and transformation framework. It defines the `Graph`/`Node` IR, proxy-based symbolic tracing, `GraphModule` (an `nn.Module` backed by a `Graph`), and an interpreter/pass infrastructure for graph transformations.

## Key Files

| File | Purpose |
|---|---|
| `torch/fx/graph.py` | `Graph`: ordered list of `Node`s; `CodeGen` emits Python source from the graph; `PythonCode` holds generated code |
| `torch/fx/node.py` | `Node`: single operation (placeholder, call_function, call_method, call_module, get_attr, output); stores op, target, args, kwargs, meta |
| `torch/fx/graph_module.py` | `GraphModule`: `nn.Module` subclass backed by a `Graph`; `recompile()` regenerates `forward()` source from the graph |
| `torch/fx/_symbolic_trace.py` | `Tracer`: proxy-based symbolic tracer; `symbolic_trace(module)` wraps calls through `Proxy` to record them as `Node`s |
| `torch/fx/proxy.py` | `Proxy`: wraps a `Node` and intercepts attribute/call operations to record them into the graph |
| `torch/fx/interpreter.py` | `Interpreter`: executes a `GraphModule` node-by-node; base class for analysis passes |
| `torch/fx/passes/` | Graph transformation passes: dead-code elimination, shape propagation, operator fusion |

## Public Interface

`torch.fx.symbolic_trace(root)`, `torch.fx.Graph`, `torch.fx.Node`, `torch.fx.GraphModule`, `torch.fx.Proxy`, `torch.fx.Tracer`, `torch.fx.Interpreter`, `torch.fx.Transformer`, `torch.fx.subgraph_rewriter.replace_pattern()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo's `OutputGraph` uses `SubgraphTracer` (extends `Tracer`) to build FX graphs |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor's lowering pass reads `GraphModule` IR |
| `torch._C` | depends-on | `_fx_map_arg`, `_NodeIter` (imported in `graph.py`) for fast node traversal |

## Runtime Behaviour

`symbolic_trace(module)` wraps the module's `forward` inputs in `Proxy` objects and calls `forward`; each `torch.*` / `nn.Module` call intercepted by `Proxy.__getattr__` and `__call__` records a `Node` into the `Graph`. After tracing, `GraphModule(module, graph)` calls `recompile()`, which uses `CodeGen` to emit a Python `forward` function as a string and `exec`-compiles it. `Interpreter.run()` walks the node list in topological order, dispatching each `call_function`/`call_module`/`call_method` node to its concrete callable. Graph mutations (adding, removing, or relinking nodes) update `Node.users` and `Node._prev`/`Node._next` linked-list pointers.

## Performance Profile

`Graph` stores nodes in a doubly-linked list (using `Node._prev`/`Node._next` from `_C._NodeIter`), so insertion and deletion are O(1). `recompile()` uses `exec` on generated Python source, which has one-time JIT compilation cost paid on first `GraphModule` construction or graph mutation; subsequent calls hit the compiled bytecode. `symbolic_trace` has overhead proportional to the number of ops traced — every operation creates a `Proxy` and records a `Node`. Passes over the graph (shape propagation, DCE) are O(nodes); the `passes/` directory contains implementations that can be composed.

## Design Rationale

`Graph`/`Node` as a plain Python data structure (rather than a C++ IR) keeps transformation passes writable in pure Python and inspectable via `print_tabular()` / `print_readable()`. `Proxy`-based tracing reuses the actual module `forward` code as the trace specification — no separate trace DSL is needed. `GraphModule.recompile()` generates Python source rather than bytecode directly so the resulting `forward` is human-readable and debuggable with standard Python tools. The `Interpreter` base class enables analysis-without-execution and controlled re-execution passes without reimplementing dispatch logic.
