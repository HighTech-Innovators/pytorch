# `torch/fx`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/fx` provides PyTorch's Python-level graph IR, symbolic tracer, generated-code module wrapper, and graph interpreter, as mapped in book chapter 09. It represents programs as `Graph` objects containing linked `Node` objects, packages graphs as executable `GraphModule` modules, and provides `Proxy`/`Tracer` machinery for symbolic tracing. Dynamo, export, AOTAutograd, Inductor, quantization, and user graph transforms use FX as their shared Python-native interchange format.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public FX namespace; exports `symbolic_trace`, `Tracer`, `Graph`, `GraphModule`, `Interpreter`, `Transformer`, `Node`, `Proxy`, and `replace_pattern` |
| `graph.py` | `Graph` container, node insertion/erasure APIs, name generation, Python code generation, and pytree input/output codegen |
| `node.py` | `Node` data model, legal op names, target/argument types, user tracking, side-effect markers, and argument mapping helpers |
| `graph_module.py` | `GraphModule` class that owns a `Graph`, compiles generated `forward` code, supports deepcopy, packaging, and deserialization |
| `_symbolic_trace.py` | `Tracer` and `symbolic_trace` implementation that run Python with proxies and record operations into a graph |
| `proxy.py` | `Proxy`, `TracerBase`, `GraphAppendingTracer`, scope tracking, and magic-method interception used during tracing |
| `interpreter.py` | Node-by-node `Interpreter` and `Transformer` for execution, analysis, and graph rewrites |
| `subgraph_rewriter.py` | Pattern replacement utilities that match and rewrite FX subgraphs |
| `passes/` | Reusable graph passes for shape propagation, runtime asserts, splitting, and other transformations |

## Public Interface

`torch.fx.symbolic_trace()` captures an `nn.Module` or callable into a `GraphModule`. `Graph` exposes construction and mutation APIs such as `placeholder`, `call_function`, `call_method`, `call_module`, `get_attr`, `output`, insertion contexts, `erase_node`, and dead-code elimination. `Node` exposes `op`, `target`, `args`, `kwargs`, `users`, `meta`, replacement helpers, and formatting. `GraphModule` exposes executable `forward`, `.graph`, `.code`, `recompile()`, serialization support, and module attribute lookup. `Interpreter` executes graph nodes with overridable `placeholder`, `get_attr`, `call_function`, `call_method`, `call_module`, and `output` hooks; `Transformer` uses the same structure to rewrite graphs.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | `GraphModule` subclasses `nn.Module`, traces submodules, and stores parameters/buffers as module attributes |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo emits FX graphs and consumes `GraphModule` for backend compilation |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor lowers FX nodes and reads node metadata for scheduling and codegen |
| [torch/_export](torch/_export/ADR.md) | depended-on-by | Export stores `ExportedProgram.graph_module` as FX and verifies the FX graph contract |
| [torch/_functorch](torch/_functorch/ADR.md) | depended-on-by | AOTAutograd partitions, transforms, and recompiles FX graphs |
| [torch/utils](torch/utils/ADR.md) | depends-on | Pytree utilities flatten structured inputs/outputs and rebuild them in generated code |

## Runtime Behaviour

`symbolic_trace()` builds a `Tracer`, replaces tensors and module attributes with `Proxy` objects, executes user Python, and records proxy operations as `Node` objects in a `Graph`. Each node records an opcode category, a target callable or name, structured args/kwargs that can reference other nodes, a reverse `users` map, and arbitrary `meta` such as fake tensor values, stack traces, or tensor metadata. `GraphModule` turns the graph into Python source, installs it in linecache through `_exec_with_source`, compiles it into `forward`, and recompiles when the graph changes. `Interpreter.run()` walks nodes in graph order, resolves inputs from an environment, executes each operation kind, frees values after last use when enabled, and enriches errors with node and graph artifacts.

Graph mutation stays local and explicit. Passes insert nodes through `inserting_before` or `inserting_after`, redirect uses with `replace_all_uses_with`, erase dead nodes with `erase_node`, then call `gm.recompile()` so generated code matches the new graph.

## Performance Profile

FX graph manipulation favors transformation speed and debuggability over low-level execution speed. `Graph` stores nodes in an ordered linked structure, so sequential passes, insertion, deletion, and topological rewrites are efficient for compiler workflows; random access is less important than preserving program order and stable code generation. Generated `GraphModule.forward` runs as normal Python, so standalone FX execution still pays Python call overhead per node; compiler backends use FX as an IR and lower it to fused kernels when runtime speed matters. `Interpreter` can free intermediate values after their last use, which reduces analysis-time memory pressure for large graphs.

## Design Rationale

FX uses Python objects and generated Python source because PyTorch compiler passes and researchers need inspectable, hackable graphs that integrate with existing modules, debuggers, and Python callables. `GraphModule` wraps a graph in `nn.Module` so parameters, buffers, submodules, serialization, and module APIs continue to work. Nodes keep generic Python targets instead of a fixed operator enum because FX represents both ATen operators and arbitrary Python/module calls. Metadata lives on nodes rather than in a separate side table so shape, dtype, source, stack, quantization, partitioning, and backend annotations travel with the IR through passes.
