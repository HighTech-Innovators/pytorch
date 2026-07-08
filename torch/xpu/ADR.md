# `torch/xpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/xpu` owns the Intel XPU backend's Python runtime. It handles device discovery, lazy initialization, streams and events, memory management, RNG state, and graph capture for XPU execution.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Implements lazy XPU initialization, device enumeration, device guards, stream selection, and public backend queries |
| `memory.py` | Exposes allocator statistics, memory-fraction controls, snapshots, and cache management for XPU devices |
| `random.py` | Manages XPU default generators, seed APIs, and per-device RNG state |
| `streams.py` | Defines the `Stream` and `Event` wrappers over native XPU stream and event bases |
| `graphs.py` | Defines `XPUGraph`, the `graph` context manager, and `make_graphed_callables` for XPU graph capture |

## Public Interface

Important public symbols are `device_count`, `is_available`, `is_initialized`, `init`, `device`, `device_of`, `set_device`, `current_device`, `get_device_name`, `get_device_capability`, `get_device_properties`, `can_device_access_peer`, `Stream`, `Event`, `stream`, `set_stream`, `current_stream`, `empty_cache`, `memory_stats`, `mem_get_info`, `set_per_process_memory_fraction`, `get_rng_state`, `manual_seed_all`, `XPUGraph`, `graph_pool_handle`, `graph`, and `make_graphed_callables`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/accelerator](torch/accelerator/ADR.md) | depended-on-by | the generic accelerator facade delegates availability, device counting, stream control, RNG, and memory operations to backend modules such as `torch.xpu` |
| [torch/numa](torch/numa/ADR.md) | mutual | NUMA binding computes CPU affinity from accelerator-local device indices, and XPU provides the device identity the binding layer reasons about |
| [torch/compiler](torch/compiler/ADR.md) | depends-on | graph capture code interoperates with compiler settings such as `force_cudagraph_gc` through the shared accelerator graph model |

## Runtime Behaviour

`__init__.py` keeps XPU initialization lazy: `_lazy_call` queues deferred operations, `_lazy_init()` populates `default_generators`, and `_is_in_bad_fork` blocks unsafe forked initialization paths. `_parse_visible_devices()` reads `ZE_AFFINITY_MASK`, and `_enum_zes_device_infos()` queries Level Zero Sysman through `pyzes` to populate `_cached_zes_device_infos`, preferring visible discrete GPUs over integrated GPUs when both are present. `memory.py` exposes allocator state by calling native helpers like `torch._C._xpu_memoryStats`, `torch._C._xpu_getMemoryInfo`, and `torch._C._xpu_setMemoryFraction`. `random.py` routes seeding and RNG-state updates through `_lazy_call`, while `graphs.py` uses `XPUGraph.capture_begin()` and the `graph` context manager to synchronize, clear cache, switch streams, and capture replayable XPU work.

## Performance Profile

Lazy initialization avoids paying runtime setup and generator construction costs on import, which matters for processes that may inspect XPU availability without using the backend. Device enumeration caches Level Zero information in `_cached_zes_device_infos`, so repeated `device_count()` and property lookups do not re-scan Sysman state unless the process is restarted. `memory_stats()` flattens a nested native dictionary into an ordered Python mapping, which allocates on each call and is best treated as a diagnostics API rather than a hot-path primitive. XPU graph capture reduces launch overhead by replaying pre-recorded work and can reuse graph memory pools across callables, but capture intentionally synchronizes and empties cache to make the recorded allocation pattern stable.

## Design Rationale

The backend owns the full Python control plane for XPU because device enumeration, lazy runtime setup, and stream semantics are backend-specific even when higher-level APIs look similar to CUDA. Keeping memory, RNG, streams, and graphs under one namespace makes XPU usable as a first-class backend rather than a thin alias over generic accelerator hooks.
