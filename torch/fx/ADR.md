# `torch/fx`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/fx` provides the FX intermediate representation (IR): a Python-level graph of operations over tensors, a tracer that captures module calls symbolically, and an interpreter/transformer framework for graph analysis and rewriting. It is the IR substrate that TorchDynamo, TorchInductor, and export pipelines use to represent compiled programs.

## Key Files

| File | Purpose |
|---|---|
| `graph.py` | `Graph` — container of `Node` objects; `PythonCode` generator; legal ops: `call_function`, `call_method`, `get_attr`, `call_module`, `placeholder`, `output` |
| `node.py` | `Node` — a single operation in the graph; `op`, `name`, `target`, `args`, `kwargs`, `users`, `meta`; `map_arg` traversal utility |
| `graph_module.py` | `GraphModule` — executable `nn.Module` whose `forward` is synthesised from a `Graph`; `recompile()` regenerates the forward code string |
| `_symbolic_trace.py` | `Tracer` — symbolic execution tracer; overrides `__torch_function__` to record operations into a `Graph`; `symbolic_trace(module)` entry point |
| `proxy.py` | `Proxy` — wraps a `Node`; overrides Python operators to produce new `Node`s during tracing |
| `interpreter.py` | `Interpreter` — reference graph executor; `run(input)` evaluates each node in topological order using real tensors |
| `passes/` | Graph transformation passes: `eliminate_dead_code`, `shape_prop`, `operator_schemas` |
| `subgraph_rewriter.py` | Pattern-based subgraph matching and replacement |
| `_pytree.py` | FX-local pytree utilities for flattening/unflattening node arguments |
| `immutable_collections.py` | `immutable_dict`, `immutable_list` — used for `Node.args` and `Node.kwargs` to detect mutations |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.fx.symbolic_trace(module)` | Returns a `GraphModule` capturing `module.forward` as a `Graph` |
| `torch.fx.Graph` | IR container; `graph.nodes` iterates `Node` objects in topological order; `graph.create_node`, `graph.erase_node`, `graph.output` |
| `torch.fx.Node` | Single operation; `node.op` is one of the six legal ops; `node.args`/`node.kwargs` reference other nodes or literal values |
| `torch.fx.GraphModule` | Executable module from a `Graph`; `gm.code` shows the generated Python source |
| `torch.fx.Proxy` | Tracing wrapper; arithmetic operators record operations into the active graph |
| `torch.fx.Tracer` | Customisable tracer; override `is_leaf_module`, `call_module`, `call_function` to control capture |
| `torch.fx.Interpreter` | Evaluates a `GraphModule` node-by-node with real tensors; override `run_node` to add instrumentation |
| `torch.fx.Transformer` | `Interpreter` subclass that rebuilds the graph with modifications |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | `symbolic_trace` wraps `nn.Module`; `call_module` nodes reference module attributes |
| `torch._C._fx_map_arg` | depends-on | C++ `map_arg` traversal used in `node.py` for fast argument tree walking |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo uses `torch.fx.Graph` as the output IR of bytecode analysis |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor receives a `GraphModule` from Dynamo and lowers it to Triton/C++ kernels |
| [torch/_functorch](torch/_functorch/ADR.md) | depended-on-by | AOT autograd uses `make_fx` to trace the forward+backward as a single `Graph` |
| [torch/_export](torch/_export/ADR.md) | depended-on-by | Export serialises `GraphModule` to ONNX / flatbuffer format |

## Runtime Behaviour

`symbolic_trace(module)` installs the `Tracer` and runs `module.forward` with `Proxy` inputs. Every operation on a `Proxy` calls `Tracer.create_proxy`, which creates a `Node` in the `Graph` and returns a new `Proxy` wrapping it. After tracing completes, `GraphModule.__init__` calls `recompile()`, which uses `Graph.python_code()` to generate a Python function string and `exec()` it to create the live `forward` method. The generated code is a flat sequence of assignments with no Python control flow (only `torch.*` operations and module attribute reads). `Interpreter.run(inputs)` evaluates the graph node-by-node by maintaining an `env` dict from node to value and calling `run_node` for each operation.

## Performance Profile

- **Allocation sites**: `symbolic_trace` allocates one `Node` per operation during tracing; for a typical ResNet-50 this is ~150 nodes. `GraphModule.recompile()` calls `exec()` to compile the generated string; this is a one-time cost at trace time.
- **Synchronization costs**: none — FX is a pure Python data structure manipulation layer with no threading or GPU synchronisation.
- **Data movement**: `symbolic_trace` runs with `Proxy` objects (no real tensor data); no data movement occurs during tracing. `Interpreter.run` executes with real tensors and triggers normal ATen dispatch.
- **Redundant or repeated work**: each `graph.eliminate_dead_code()` pass does a reverse BFS over all nodes; for graphs with many dead nodes this can be called multiple times by the Inductor pipeline. `shape_prop` runs a full `Interpreter` pass to populate `node.meta['val']` with fake-tensor shapes.

## Design Rationale

`Graph` stores `Node` objects in a doubly-linked list rather than a Python list so that `insert_before` and `erase_node` are O(1) without index shifting. `Node.args` and `Node.kwargs` are `immutable_dict` / `immutable_list` instances so that mutations are detected immediately — a safeguard against transformations that break data-flow edges by mistake. `Proxy` overrides Python's magic methods (`__add__`, `__matmul__`, `__getattr__`, etc.) so that tracing works on unmodified user code without AST rewriting. `GraphModule` uses `exec` to produce the forward function rather than an interpreter loop so that the compiled module runs at Python interpreter speed rather than paying per-node dispatch overhead.
