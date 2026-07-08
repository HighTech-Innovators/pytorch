# `torch/_native`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_native` owns Python-level native DSL operator overrides for ATen. It lets DSL backends register conditional or unconditional replacements for specific dispatcher keys, exposes an opt-in decomposition table for compile paths, and instruments first-call kernel compilation.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Imports DSL utility modules, resolves optional user graph ordering, and triggers `_register_all_overrides()` at import time. |
| `registry.py` | Stores override graphs, installs eager routers and compile routers, and exposes registration and filtering APIs. |
| `dsl_registry.py` | Tracks registered DSL modules, their availability, and version queries. |
| `instrumentation.py` | Emits human-readable and structured compile/cache events for CuTeDSL and Triton kernel compilation. |

## Public Interface

The main APIs are `register_op_override()`, `native_decomp_table()`, `reorder_graphs_from_user_function()`, `reenable_op_overrides()`, `deregister_op_overrides()`, and `get_dsl_operations()`. Registration state is represented by `_OverrideNode`, `_FilterState`, and the global `dsl_registry`, while instrumentation exports `CompileEvent`, `instrument_cutedsl_compile()`, and `instrument_triton_kernel()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_library](torch/_library/ADR.md) | depends-on | `registry.py` defines `_native::<id>` operators and ATen override routers with `torch.library.Library` and `torch.library.get_kernel()`. |
| [torch/_logging](torch/_logging/ADR.md) | depends-on | `instrumentation.py` checks structured tracing state in `trace_log.handlers` and emits compile artifacts through `trace_structured()`. |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Compile consumers thread `native_decomp_table()` into Dynamo or export so native overrides participate in graph lowering. |

## Runtime Behaviour

`register_op_override()` does not mutate the dispatcher immediately; it appends an `_OverrideNode` to the per-`(op_symbol, dispatch_key)` graph in `_graphs`, assigns a stable `_native::<node_id>` operator name, and records secondary maps for later filtering and deregistration. When registration is materialized, `_register_overrides_from_graph()` defines or reuses the `_native` library, installs each node implementation, captures the prior ATen kernel with `torch.library.get_kernel()`, and builds an eager router plus a compile router that both use first-match predicate dispatch.

`native_decomp_table()` returns either the raw `_native_decomp_overrides` or those overrides layered on top of `torch.export.default_decompositions()`, which is the compile-time entry point for explicit opt-in routing. `DSLRegistry.register_dsl()` validates the module interface, stores the module by name, and clears its cached availability and version queries, while `instrument_cutedsl_compile()` and `instrument_triton_kernel()` time compile-capable call sites and emit `CompileEvent` payloads only when logging or structured tracing is enabled.

## Performance Profile

- **Allocation sites** - Import-time registration allocates override graph nodes, dispatcher libraries, and router closures, while instrumentation allocates a `CompileEvent` only on calls where `_listening()` sees an active logger or trace sink.
- **Synchronization costs** - Eager routers avoid explicit device synchronization and simply fall back to the captured ATen kernel when no predicate matches, but first-match predicate evaluation still adds Python dispatch overhead on every overridden call.
- **Data movement** - The registry mostly rewires control flow instead of moving data; compile routers return `NotImplemented` on a miss so Inductor or export can reuse the default lowering without materializing alternate tensors.
- **Redundant or repeated work** - `DSLRegistry` caches `is_dsl_available()` and `get_dsl_version()`, and `_register_overrides_from_graph()` captures the fallback kernel once per `(op, key)` so later calls do not need to rediscover the native implementation.

## Design Rationale

Representing overrides as editable graphs instead of one-shot registrations makes reordering, selective disablement, and compile-time opt-in decomposition practical. The explicit `native_decomp_table()` API also avoids the global side effects that would come from patching compiler decomposition tables during `import torch._native`.
