# `torch/_lazy`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_lazy` is the Python control layer for lazy tensor execution backends. It exposes step boundaries, queued post-step closures, graph hash helpers, and a wrapper that extracts and replays cached lazy graphs without retracing.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Exposes step control APIs such as `mark_step()`, `wait_device_ops()`, `sync_multi()`, and `to_cpu()`. |
| `closure.py` | Implements synchronous and asynchronous step-closure queues, including `AsyncClosureHandler`. |
| `device_context.py` | Maintains per-device `DeviceContext` objects that hold closure queues and handlers. |
| `computation.py` | Wraps backend-specific graph hash, graph input, and cached graph replay hooks. |
| `extract_compiled_graph.py` | Traces an FX module through lazy execution, extracts the cached graph, and returns a replay wrapper. |

## Public Interface

The directory exports `mark_step`, `wait_device_ops`, `sync_multi`, `get_tensor_id`, `to_cpu`, and `save` from `__init__.py`. Other entry points used by Dynamo and backend integrations are `add_step_closure()`, `run_step_closures()`, `extract_compiled_graph()`, `get_tensors_ts_device_data_node()`, `get_graph_hash()`, and `run_cached_graph()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | `extract_compiled_graph.py` expects an `fx.GraphModule`, rewrites graph nodes in `force_lazy_device()`, and recompiles the FX graph after device rewriting. |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | `extract_compiled_graph()` states that the returned wrapper relies on Dynamo guards to ensure replay only happens for safe input shapes and metadata. |

## Runtime Behaviour

`mark_step()` calls `torch._C._lazy._mark_step(device, [], wait=wait)` to flush the current lazy graph, then immediately runs any queued closures through `run_step_closures()`. `add_step_closure()` stores lambdas on the current `DeviceContext`, and `AsyncClosureHandler.start_event_loop()` creates a background `threading.Thread` that drains the queue until it times out and finds no more pending closures.

`extract_compiled_graph()` deep-copies the incoming `fx.GraphModule`, moves example inputs and the copied module to the `lazy` device, rewrites eager factory-device arguments in `force_lazy_device()`, and executes one lazy trace to populate metrics and backend caches. It then uses `computation.get_tensors_ts_device_data_node()`, `computation.get_graph_hash()`, and `lazy.sync_multi()` to build a `GraphInputMatcher`, capture the cached graph hash, and return `optimized_mod()`, which replays `computation.run_cached_graph()` and copies mutated outputs back into the original input tensors when necessary.

## Performance Profile

- **Allocation sites** - `extract_compiled_graph()` allocates a deep-copied FX module, lazy copies of every example input, graph-input match tables, and a `ReturnValueHandler` to expand deduplicated lazy outputs back to Python-level output structure.
- **Synchronization costs** - `mark_step()` is the explicit barrier for lazy execution, and `to_cpu()` forces materialization by calling `sync_multi()` before moving every flattened tensor back to CPU.
- **Data movement** - Replay paths avoid retracing, but `optimized_mod()` still reconstructs graph inputs, receives eager outputs from `run_cached_graph()`, duplicates aliased outputs, and may perform `arg.copy_(res[i])` for mutated arguments.
- **Redundant or repeated work** - Per-device `DeviceContext` objects and cached graph hashes avoid rebuilding closure handlers or recompiling identical graphs, while fallback-op counting in `get_fallback_ops()` intentionally scans metrics counters only during extraction time.

## Design Rationale

The directory keeps backend-specific lowering and execution inside `torch._C._lazy` and `torch._C._lazy_ts_backend`, while Python owns user-facing orchestration and FX graph rewriting. That split lets the lazy runtime stay backend-centric and fast, while higher-level tools can still reason about steps, closures, and cached graph replay from Python.
