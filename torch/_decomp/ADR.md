# `torch/_decomp`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_decomp` owns the operator decomposition registry: a set of tables mapping `torch.ops.aten.*` operator overloads to Python functions that express each op in terms of simpler primitives. The compiler stack (AOTAutograd, Inductor, `torch.export`) queries these tables to lower complex ATen ops into sequences that downstream backends can implement natively.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Registry infrastructure: `global_decomposition_table` (keyed `post_autograd`, `pre_autograd`, `meta`), `register_decomposition` decorator, `get_decompositions`, `remove_decompositions`, `core_aten_decompositions` |
| `decompositions.py` | ~6,300-line library of 177+ `@register_decomposition`-decorated functions mapping ATen ops to primitive equivalents (e.g., `aten.layer_norm` → `aten.mean` + `aten.var` + `aten.sub` + `aten.mul` + `aten.add`) |
| `decompositions_for_jvp.py` | JVP-specific decomposition table (`decomposition_table_for_jvp`); extends the post-autograd table with forward-mode gradient rules |
| `decompositions_for_rng.py` | RNG op decompositions (`rng_decompositions`); splits `aten.rand*` and `aten.dropout` into offset-and-sample forms suitable for graph-level checkpointing |

## Public Interface

| Symbol | Purpose |
|---|---|
| `decomposition_table` | Dict mapping `OperatorBase → Callable`; post-autograd decompositions used by AOTAutograd and Inductor |
| `pre_autograd_decomposition_table` | Pre-autograd decompositions; lowered before backward graph construction |
| `meta_table` | Meta-device implementations; used by `FakeTensor` for shape inference |
| `register_decomposition(aten_op, *, type, unsafe)` | Decorator registering a Python function as the decomposition for one or more ATen ops; `type` selects the target table |
| `get_decompositions(aten_ops, type)` | Returns a subset dict of decompositions for the provided op list; consumed by `torch._functorch` compilers |
| `remove_decompositions(decompositions, aten_ops)` | Removes selected ops from a previously built decomposition dict |
| `core_aten_decompositions()` | Returns the default export decomposition set via `torch.export.exported_program.default_decompositions` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depended-on-by | `compilers.py` calls `get_decompositions`; `predispatch.py` reads `decomposition_table` directly |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor compile path queries decompositions to lower ATen ops before code generation |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | FX graph passes apply decompositions as graph-level rewrite rules, replacing complex op nodes with primitive subgraphs |
| `torch._prims` | depends-on | `decompositions.py` imports `torch._prims` to express decompositions in terms of prim ops |
| `torch._prims_common` | depends-on | Type promotion utilities (`ELEMENTWISE_TYPE_PROMOTION_KIND`, `elementwise_dtypes`) used by `type_casts` wrapper and individual decompositions |
| `torch._ops` (`OperatorBase`, `OpOverload`, `OpOverloadPacket`) | depends-on | Registry keys; `_add_op_to_registry` dispatches on these types to enumerate overloads |

## Runtime Behaviour

`global_decomposition_table` is a module-level `defaultdict(dict)` initialized at import time; the three sub-tables (`post_autograd`, `pre_autograd`, `meta`) are aliases into it. When `decompositions.py` is imported (triggered by the `import torch._decomp.decompositions` line at the bottom of `__init__.py`), each `@register_decomposition`-decorated function calls `_add_op_to_registry`, which invokes `torch._C._fake_dispatch_register_decomp` or `torch._C._fake_dispatch_register_meta` to mirror the registration into the C++ `FakeTensor` dispatch tables. The registry is write-once per process: `_add_op_to_registry` raises `RuntimeError` on duplicate registration. `get_decompositions` performs a linear scan over the registry to match the caller-supplied op list, constructing the subset dict at call time. There is no concurrency protection on the registry — registration is expected to complete during module initialization before multi-threaded inference begins.

## Performance Profile

The decomposition tables themselves are Python dicts; lookup is O(1) by `OperatorBase` identity. Table construction during `import torch._decomp.decompositions` is a one-time cost: ~177 decorator invocations each executing a C++ cross-language call (`_fake_dispatch_register_decomp`). At runtime, decompositions run inside the compiler trace, not on the hot execution path — they execute once per unique graph, not once per inference call. The `type_casts` wrapper in `decompositions.py` performs `pytree.arg_tree_leaves` traversal and two `tensor.to(dtype)` casts per decomposed call, which is negligible relative to graph compilation latency. The `rng_decompositions` dict uses an additional level of nesting (split by op name) for compatibility with the checkpointing split mechanism, adding one extra dict lookup per RNG op encountered during compilation.

## Design Rationale

Decompositions are stored as plain Python dicts of `OperatorBase → Callable` rather than registered as C++ kernels so that the compiler stack can select, compose, and override them without modifying the dispatcher. The three-table split (`post_autograd`, `pre_autograd`, `meta`) allows the same operator to have different lowering strategies depending on the compilation context: a `meta` entry powers `FakeTensor` shape inference, a `pre_autograd` entry lets the compiler lower before the autograd graph is constructed, and the default `post_autograd` entry covers inference-only backends. The `unsafe` flag on `register_decomposition` bypasses the `_convert_out_params` wrapper for cases where the decomposition deliberately manages `out=` parameters itself.
