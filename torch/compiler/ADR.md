# `torch/compiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/compiler` owns the public Python compiler facade. It gives users stable entry points for `torch.compile`-related control, graph-inclusion annotations, compilation-state queries, and portable cache artifact export and import.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Exposes the user-facing compiler API including `compile`, `reset`, `allow_in_graph`, `nonstrict_trace`, backend selection, and cache helpers |
| `_cache.py` | Defines `CacheArtifact`, `CacheInfo`, `CacheArtifactRecorder`, and `CacheArtifactManager` for serializing portable compile caches |
| `config.py` | Defines cross-cutting compiler configuration entries such as `job_id`, `dynamic_shapes`, `force_disable_caches`, and `compile_on_one_rank` |

## Public Interface

`compile`, `reset`, `allow_in_graph`, `nonstrict_trace`, `substitute_in_graph`, `list_backends`, `disable`, `set_default_backend`, `get_default_backend`, `set_stance`, `set_enable_guard_collectives`, `cudagraph_mark_step_begin`, `wrap_numpy`, `is_compiling`, `is_dynamo_compiling`, `is_exporting`, `save_cache_artifacts`, and `load_cache_artifacts` are the primary entry points. The module also exports `config`, whose symbols include `job_id`, `dynamic_shapes`, `automatic_dynamic_shapes`, `force_disable_caches`, `compile_on_one_rank`, and `force_cudagraph_gc`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | `compile()` forwards to `torch.compile`, while `reset()`, `allow_in_graph()`, and `nonstrict_trace()` call `torch._dynamo` helpers directly |
| [torch/_inductor](torch/_inductor/ADR.md) | depends-on | cache artifacts and backend configuration are designed to carry Inductor outputs and disk-cache state across processes |
| [torch/func](torch/func/ADR.md) | depends-on | `config.py` exposes compilation settings that also affect functorch-based tracing and transforms across the stack |

## Runtime Behaviour

`compile()` is intentionally a one-line shim that forwards all arguments to `torch.compile`, so the public namespace can evolve without moving the underlying implementation. `reset()` clears in-process compiler state by calling `torch._dynamo.reset()`, which drops Dynamo caches but leaves filesystem caches alone. `allow_in_graph()` and `nonstrict_trace()` return decorators from `torch._dynamo` that change how Dynamo records specific call sites in the FX graph. `_cache.py` records cache artifacts through `CacheArtifactRecorder.record()`, deduplicates them with `CacheArtifactManager._seen_artifacts`, and serializes them through `AppendingByteSerializer` so another process can hot-load the same compiled assets.

## Performance Profile

The facade functions are deliberately thin; almost all execution cost comes from Dynamo tracing, AOTAutograd, and backend code generation rather than this package. `CacheArtifactManager` reduces repeated work by storing each `CacheArtifact` only once in `_seen_artifacts`, which avoids serializing identical entries multiple times during a process lifetime. The portable cache path still incurs byte-copy and deserialization cost when `serialize()` or `deserialize()` is requested, so it is pay-as-you-go rather than always on. Configuration reads are cheap because `config.py` installs env-backed `Config` descriptors instead of re-parsing environment variables at each call site.

## Design Rationale

The package isolates user-visible compiler controls from internal packages such as `torch._dynamo` and `torch._inductor`, which lets the public API stay stable while implementations change. Cache recording lives next to the facade because cache portability is a compiler-wide concern that cuts across tracing, lowering, and code generation.
