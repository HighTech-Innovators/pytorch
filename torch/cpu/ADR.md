# `torch/cpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/cpu` owns the CPU-side compatibility surface used by device-agnostic Python code. It reports host instruction-set capabilities and provides stub stream and event objects so generic accelerator code can run unchanged on CPU.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Implements capability queries, no-op stream and event abstractions, and CPU device helpers such as `current_device` and `device_count` |
| `amp/__init__.py` | Marks the CPU AMP namespace as part of the public package layout |

## Public Interface

`get_capabilities`, `_is_avx2_supported`, `_is_avx512_supported`, `_is_vnni_supported`, `_init_amx`, `is_available`, `synchronize`, `Stream`, `Event`, `current_stream`, `StreamContext`, `stream`, `device_count`, `set_device`, `current_device`, and `is_initialized` are the exported symbols. `get_capabilities()` returns architecture keys such as `avx2`, `avx512_f`, `amx_tile`, `sve`, and `architecture` through an immutable `MappingProxyType`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/accelerator](torch/accelerator/ADR.md) | depended-on-by | generic accelerator code expects CPU to provide `device_count`, `current_stream`, `Stream`, and `Event` with the same names as accelerator backends |
| [torch/csrc/dynamo](torch/csrc/dynamo/ADR.md) | depends-on | capability queries and thread-name helpers call `torch._C._cpu._get_cpu_capability`, `torch._C._cpu._init_amx`, `torch._C._set_thread_name`, and `torch._C._get_thread_name` |

## Runtime Behaviour

`get_capabilities()` is `@lru_cache(None)` and returns `MappingProxyType(torch._C._cpu._get_cpu_capability())`, so the expensive hardware feature probe runs once per process. `Stream` and `Event` deliberately implement no-op methods, while `_default_cpu_stream` and `_current_stream` keep enough mutable state for context-managed code to switch streams on CPU without breaking generic control flow. `StreamContext.__enter__()` and `__exit__()` swap the module-global `_current_stream`, and `current_stream()` simply returns that object. `current_device()` always returns the literal string `"cpu"`, `device_count()` always returns `1`, and `is_initialized()` always returns `True`.

## Performance Profile

The module avoids almost all runtime cost after import because capability detection is cached and stream operations are no-ops. `get_capabilities()` allocates one proxy mapping and reuses it, which keeps repeated feature checks cheap in hot code paths. There is no synchronization or device-switch overhead because `synchronize()` and `set_device()` intentionally do nothing on CPU. The stub `StreamContext` mutates a single Python global instead of consulting thread-local runtime state, so its overhead stays small but it does not model concurrent hardware queues.

## Design Rationale

The package exists so higher-level code can call `torch.cpu` with the same names it uses for accelerators without special cases. It keeps CPU semantics explicit: real capability queries come from `torch._C._cpu`, while stream and device controls are placeholders that preserve API shape rather than simulate missing hardware behavior.
