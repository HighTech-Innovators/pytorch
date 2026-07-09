# `torch/accelerator`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/accelerator` provides a device-generic Python facade for the currently compiled accelerator backend. It exposes availability, device selection, streams, synchronization, memory accounting, graph capture, and RNG state without requiring callers to hardcode `torch.cuda`, `torch.xpu`, or another backend module.

## Key Files

| File | Purpose |
|---|---|
| `torch/accelerator/__init__.py` | Defines the public accelerator API, delegates availability to backend modules, and wraps C++ device, stream, and synchronization calls |
| `torch/accelerator/memory.py` | Exposes allocator cache control, flattened memory statistics, peak resets, memory info, and allocator snapshots |
| `torch/accelerator/graphs.py` | Defines `Graph`, a Python wrapper around `torch._C._acceleratorGraph` with capture, replay, memory-pool, and debug-dump methods |
| `torch/accelerator/random.py` | Exposes default accelerator generator seed and RNG state helpers |
| `torch/accelerator/_utils.py` | Normalizes device arguments with `_get_device_index` and delegates lazy calls to the current backend module |

## Public Interface

| Symbol | Description |
|---|---|
| `current_accelerator(check_available=False)` | Returns the compile-time accelerator as `torch.device` via `torch._C._accelerator_getAccelerator()` |
| `device_count()` / `is_available()` | Delegate to the current backend module returned by `torch.get_device_module` |
| `current_device_index()`, `set_device_index()`, `device_index` | Query, set, and temporarily exchange the current accelerator device index |
| `current_stream()`, `set_stream()`, `synchronize()` | Wrap C++ stream access and device synchronization for the current accelerator |
| `get_device_capability()` | Returns backend capability data from `torch._C._accelerator_getDeviceCapability` |
| `empty_cache`, `empty_host_cache`, `memory_stats`, `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `max_memory_reserved`, `reset_accumulated_memory_stats`, `reset_peak_memory_stats`, `get_memory_info` | Allocator inspection and cache-control helpers from `memory.py` |
| `Graph` | Captures and replays accelerator work through `_acceleratorGraph` |
| `random.initial_seed`, `random.get_rng_state`, `random.get_rng_state_all` | Default generator state helpers for the current accelerator |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses `torch.device`, `torch.Stream`, `torch.Event`, `torch.Generator`, `torch.get_device_module`, backend modules, and `torch.compiler.config` |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Calls `_accelerator_getAccelerator`, `_accelerator_getDeviceIndex`, `_accelerator_setDeviceIndex`, `_accelerator_getStream`, `_accelerator_synchronizeDevice`, allocator-stat APIs, and `_acceleratorGraph` |
| [c10/core](c10/core/ADR.md) | depends-on | Relies on c10 device, stream, generator, and allocator abstractions exposed through Python bindings |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | Reuses CUDA-style caching allocator, graph, stream, and memory snapshot concepts when the current accelerator is CUDA-compatible |

## Runtime Behaviour

`current_accelerator` asks the C++ binding for the compile-time accelerator and optionally calls `is_available()` for runtime validation. `device_count()` and `is_available()` intentionally delegate to `torch.get_device_module(acc)` so backend-specific Python checks, including CUDA's non-poisoning availability path, stay in the backend module. Memory helpers normalize device arguments through `_get_device_index`, return empty statistics when the allocator is not initialized, flatten nested C++ allocator stats into a sorted `OrderedDict`, and call C++ reset or cache APIs only after validating allocator state.

## Performance Profile

Most top-level functions are thin Python wrappers around C++ accelerator calls, so steady-state overhead is argument normalization and a single binding transition. `memory_stats` performs extra Python work by recursively flattening nested stats and sorting the flattened keys before returning an `OrderedDict`. `Graph.__enter__` synchronizes the accelerator, optionally runs `gc.collect()` when `torch.compiler.config.force_cudagraph_gc` is set, empties device and host caches, and then starts capture; replay uses the instantiated backend graph to reduce repeated launch overhead.

## Design Rationale

`torch/accelerator` gives users and higher-level PyTorch subsystems one stable API for the active accelerator instead of branching across backend-specific modules. The design keeps backend discovery and policy in Python while delegating device index, stream, allocator, RNG, and graph primitives to shared C++ bindings. The separate `memory.py`, `graphs.py`, `random.py`, and `_utils.py` files keep public categories focused while sharing the same `_get_device_index` normalization rules.
