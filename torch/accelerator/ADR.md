# `torch/accelerator`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/accelerator` owns the device-agnostic accelerator facade for Python. It gives callers one namespace for capability queries, device selection, memory statistics, RNG state, and graph capture while delegating backend-specific work to the active device module.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports the top-level accelerator API and delegates `device_count`, `is_available`, `current_accelerator`, `get_device_capability`, and stream helpers to the active backend |
| `memory.py` | Wraps generic accelerator allocator APIs such as `memory_stats`, `memory_allocated`, `memory_reserved`, and `get_memory_info` |
| `random.py` | Exposes accelerator-default generator state through `initial_seed`, `get_rng_state`, and `get_rng_state_all` |
| `graphs.py` | Defines `Graph`, the Python wrapper for accelerator graph capture and replay |
| `_utils.py` | Normalizes device arguments with `_get_device_index` before calls into `torch._C` |

## Public Interface

`current_accelerator`, `device_count`, `is_available`, `current_device_index`, `set_device_index`, `get_device_capability`, `current_stream`, `set_stream`, `synchronize`, `memory_stats`, `memory_allocated`, `memory_reserved`, `reset_peak_memory_stats`, `get_memory_info`, `Graph`, `random.initial_seed`, and `random.get_rng_state` form the public surface. `Graph.capture_begin`, `Graph.capture_end`, `Graph.instantiate`, `Graph.replay`, `Graph.pool`, and `Graph.debug_dump` expose graph-capture control without tying callers to CUDA- or XPU-specific namespaces.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/xpu](torch/xpu/ADR.md) | depends-on | `device_count()` and `is_available()` call `torch.get_device_module(acc)` and delegate to backend modules such as `torch.xpu` |
| [torch/compiler](torch/compiler/ADR.md) | depends-on | `graphs.py` checks `torch.compiler.config.force_cudagraph_gc` before capture begins |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | `memory.py`, `random.py`, and `__init__.py` call generic accelerator entry points in `torch._C` such as `_accelerator_getDeviceStats` and `_accelerator_getDefaultGenerator` |

## Runtime Behaviour

`current_accelerator()` calls `torch._C._accelerator_getAccelerator()` and returns a `torch.device` only when the compiled backend exists and, when requested, `is_available()` also succeeds. `device_count()` and `is_available()` then fetch the backend module with `torch.get_device_module(acc)` and forward to its Python implementation instead of duplicating backend logic in this package. `memory_stats()` calls `torch._C._accelerator_getDeviceStats(device_index)` and flattens the nested allocator report into a sorted `OrderedDict` so callers get stable string keys. `Graph.__enter__()` synchronizes the current accelerator, optionally runs `gc.collect()` under `torch.compiler.config.force_cudagraph_gc`, empties device and host caches, and only then starts capture.

## Performance Profile

The package itself adds only thin Python dispatch because the heavy work stays in backend modules and `torch._C` entry points. `memory_stats()` allocates and sorts a flattened list of statistic pairs on every call, so repeated polling costs Python time even when allocator state is unchanged. `Graph` reduces launch overhead on replay by capturing kernels once, but `Graph.__enter__()` can intentionally pay a high synchronization and garbage-collection cost to maximize reusable capture memory. `random.get_rng_state_all()` performs one generator query per visible device, so its cost scales with `torch.accelerator.device_count()`.

## Design Rationale

The package centralizes APIs that mean the same thing across accelerators and keeps backend-specific policy in device modules such as `torch.xpu`. `Graph`, `memory`, and `random` sit beside the selector functions because graph capture, allocator stats, and default generators are the recurring cross-backend control surfaces that users need in device-agnostic code.
