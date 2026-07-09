# `torch/compiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/compiler` provides the public Python namespace for compiler-facing APIs around `torch.compile`, Dynamo tracing controls, guard policies, compile stances, cache artifact hot loading, and nested compile regions. It presents stable entry points while delegating implementation to `torch._dynamo`, `torch._inductor`, `torch._functorch`, `torch._higher_order_ops`, and shared configuration machinery.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API surface for compiler controls, tracing decorators, guard filters, cache artifact save/load, and nested compile region marking |
| `config.py` | Cross-cutting compiler configuration module built from `torch.utils._config_module.Config` aliases and environment-backed options |
| `_cache.py` | Cache artifact serialization, deserialization, recording, deduplication, and cache population utilities |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.compiler.compile()` | Delegates directly to `torch.compile(*args, **kwargs)` |
| `reset()` | Calls `torch._dynamo.reset()` to clear in-process compiler state |
| `allow_in_graph()` / `nonstrict_trace()` / `substitute_in_graph()` | Register Dynamo graph inclusion, nonstrict tracing, and polyfill behavior |
| `list_backends()` | Delegates to `torch._dynamo.list_backends()` and filters debug or experimental tags by default |
| `disable()` | Delegates to `torch._dynamo.disable()` with optional recursive disabling and a reason |
| `set_default_backend()` / `get_default_backend()` | Delegate to `torch._dynamo.backends.registry` to control the default backend |
| `set_stance()` | Delegates to `torch._dynamo.set_stance()` and supports stances such as `default`, `force_eager`, `eager_on_recompile`, and `eager_then_compile` |
| `set_enable_guard_collectives()` | Installs or clears Dynamo guard collective hooks through `torch._C._dynamo.eval_frame.set_guard_complete_hook` |
| `cudagraph_mark_step_begin()` | Calls `torch._inductor.cudagraph_trees.mark_step_begin()` |
| `wrap_numpy()` | Delegates to `torch._dynamo.external_utils.wrap_numpy` |
| `is_compiling()` / `is_dynamo_compiling()` / `is_exporting()` | Report compiler, Dynamo, and export tracing state using module flags and JIT scripting checks |
| `save_cache_artifacts()` / `load_cache_artifacts()` | Serialize and hot-load compiler cache artifacts through `CacheArtifactManager` |
| `keep_portable_guards_unsafe()` and `skip_*_unsafe()` guard filters | Return per-guard booleans for portable, tensor, module, global, or all-guard filtering |
| `nested_compile_region()` | Marks a callable region with `torch._higher_order_ops.invoke_subgraph.mark_compile_region` |
| `load_compiled_function()` | Loads an AOT-compiled function from a file-like object with optional globals and external data |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Re-exports behavior around `torch.compile`, checks `torch.jit.is_scripting()`, and uses `torch.nn.Parameter` in guard filters |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Delegates reset, graph inclusion, nonstrict tracing, backend listing, disabling, stance management, guard hooks, cache precompile, and package artifacts |
| [torch/_inductor](torch/_inductor/ADR.md) | depends-on | Uses `cudagraph_trees.mark_step_begin()` and imports `InductorCacheArtifact` and `AutotuneCacheArtifact` for cache deserialization |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Uses AOTAutograd stacktrace preservation hooks and registers `AOTAutogradCacheArtifact` |
| [torch/_export](torch/_export/ADR.md) | depended-on-by | Export and compiler tracing share public state queries such as `is_compiling()` and `is_exporting()` |

## Runtime Behaviour

Most public functions import implementation modules lazily inside the function body, then delegate immediately to the lower-level subsystem. `compile()` calls `torch.compile`, `reset()` imports `torch._dynamo` and calls `torch._dynamo.reset()`, `set_stance()` calls `torch._dynamo.set_stance()` and marks itself Dynamo-forbidden, and `cudagraph_mark_step_begin()` imports `torch._inductor.cudagraph_trees` only when invoked. Context managers such as `_non_strict_tracing_context()` and `_compile_session_context()` save module-level boolean flags, set them for the duration of the `yield`, and restore them in `finally` blocks. `_cache.py` records artifacts through `CacheArtifactRecorder.record()`, deduplicates them with `_seen_artifacts`, serializes new artifacts with `AppendingByteSerializer`, and deserializes by importing every registered artifact class before calling `populate_cache()`.

## Performance Profile

The API layer minimizes import and call overhead by deferring heavy `torch._dynamo`, `torch._inductor`, and `torch._functorch` imports until a specific feature is used. Guard filter helpers operate with a single pass over `guard_entries` and return boolean lists, so their cost scales linearly with the number of guards. `config.py` centralizes options as `Config` aliases, which avoids duplicating compiler configuration state and lets existing Dynamo settings drive cross-cutting controls such as `dynamic_shapes`, `recompile_limit`, and `enable_cpp_symbolic_shape_guards`. `CacheArtifactManager.serialize()` only extends the serializer when new artifacts exist, deep-copies `CacheInfo` before returning it, and skips duplicate artifacts through `_seen_artifacts`.

## Design Rationale

The namespace provides a stable public facade while allowing compiler internals to remain split across Dynamo, Inductor, AOTAutograd, export, and higher-order-op packages. Lazy imports keep `import torch.compiler` cheap and avoid initializing optional compiler subsystems before users request them. Unsafe guard-filter APIs keep their policy functions explicit so users must opt into reduced guard coverage at `torch.compile` call sites. Cache artifact classes register themselves through `CacheArtifactFactory`, which makes the hot-load format extensible while preserving one manager for serialization, deserialization, and population.
