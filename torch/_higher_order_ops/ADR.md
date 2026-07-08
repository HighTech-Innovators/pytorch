# `torch/_higher_order_ops`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_higher_order_ops` owns Python higher-order operators that keep control flow, subgraphs, and effectful calls explicit across `torch.compile()` and `torch.export()`. It provides the dispatcher registrations, schema generation, tracing hooks, and autograd wrappers needed for operators such as `cond`, `while_loop`, and `with_effects`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Collects the exported HOP namespace, including `cond`, `while_loop`, `map`, `scan`, `with_effects`, and `BaseHOP`. |
| `base_hop.py` | Implements `BaseHOP`, shared fake, proxy, autograd, and functionalization registrations, and `BaseHOPFunction` backward handling for subgraph-style HOPs. |
| `cond.py` | Defines `CondOp`, the user-facing `cond()` API, branch tracing, fake tensor handling, and autograd integration. |
| `while_loop.py` | Defines `WhileLoopOp`, the user-facing `while_loop()` API, flattened carry handling, tracing, and fake/autograd implementations. |
| `effects.py` | Defines `WithEffects`, effect registration, token threading, and functionalization support for ordered side effects. |

## Public Interface

The package exports `cond`, `while_loop`, `with_effects`, `BaseHOP`, `scan`, `map`, `invoke_subgraph`, `flat_apply`, `foreach_map`, `flex_attention`, `flex_gemm`, `out_dtype`, `strict_mode`, `invoke_quant`, `invoke_quant_packed`, `wrap_activation_checkpoint`, `wrap_with_autocast`, `wrap_with_set_grad_enabled`, and `while_loop_stack_output`. The core operator classes visible to other subsystems are `CondOp`, `WhileLoopOp`, `WithEffects`, `BaseHOP`, and `BaseHOPFunction`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dispatch](torch/_dispatch/ADR.md) | depends-on | `base_hop.py` imports `suspend_functionalization()` to keep nested backward tracing and HOP functionalization stable. |
| [torch/_library](torch/_library/ADR.md) | depends-on | `effects.py` stores effect metadata in `torch._library.simple_registry` and recognizes `CustomOpDef` instances when threading tokens. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depends-on | `cond.py`, `while_loop.py`, and `base_hop.py` register fake implementations against `FakeTensorMode` and use functional tensor helpers. |
| [torch/fx](torch/fx/ADR.md) | depends-on | `materialize_as_graph()`, `reenter_make_fx()`, and branch tracing materialize user callables into `torch.fx.GraphModule` objects. |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo emits these operators during graph capture and relies on their schema and tracing behavior to preserve structured control flow. |

## Runtime Behaviour

`cond()` in `cond.py` validates the predicate and branch callables, executes a direct Python branch when `pred` is a constant eager boolean, and otherwise routes through `_hop_compile_and_call()` or `cond_op` so tracing keeps both branches alive. `CondOp.gen_schema()` materializes both branches with `materialize_as_graph()`, unions mutated input indices from `check_input_alias_and_mutation_return_outputs()`, and builds a dispatcher schema with `HopSchemaGenerator`.

`while_loop()` flattens carried and additional inputs with `pytree.tree_flatten()`, wraps `cond_fn` and `body_fn` into flat callables, and routes to `while_loop_op`, whose `gen_schema()` records mutated carry positions and output example values from the traced body graph. `BaseHOP.__init__()` registers Autograd, Functionalize, ProxyTorchDispatchMode, fake, and CompositeExplicitAutograd handlers up front, while `BaseHOPFunction.backward()` synthesizes a joint forward-backward graph with `create_fw_bw_graph()` and then re-invokes the HOP over a generated backward subgraph.

`WithEffects` in `effects.py` registers known effectful ops such as `aten::_print`, `call_torchbind`, and `invoke_leaf_function`, then threads dummy token tensors through `with_effects()` so later graph passes cannot reorder those operations. The Proxy and Functionalize implementations keep the token edges visible in traced graphs without changing eager results.

## Performance Profile

- **Allocation sites** - Compile-time work dominates because `cond`, `while_loop`, and `BaseHOP` materialize `GraphModule` subgraphs, flatten and unflatten pytrees, synthesize schemas, and allocate proxy nodes or token tensors during tracing.
- **Synchronization costs** - The dense eager paths such as `cond_op_dense()` and `with_effects_dense()` do not synchronize devices, but autograd and tracing paths re-enter functionalization, fake tensor, and proxy modes that add dispatcher traffic before kernels run.
- **Data movement** - `while_loop()` and `cond()` avoid moving tensor data, yet they repeatedly copy Python container structure, duplicate metadata, and in `with_effects` prepend extra token outputs that every downstream pass must carry.
- **Redundant or repeated work** - `BaseHOP` caches little by design because branch graphs and mutation signatures depend on the specific callable and operand set; the cost is paid again whenever Dynamo or export retraces a new subgraph.

## Design Rationale

The directory models structured control flow as explicit operators instead of lowering everything to opaque Python callbacks, because compilers need branch and loop boundaries to survive graph capture. Shared machinery in `BaseHOP` and `effects.py` keeps fake tensor, autograd, functionalization, and proxy behavior consistent across many operators without duplicating dispatcher setup in each file.
