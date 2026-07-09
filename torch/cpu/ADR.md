# `torch/cpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/cpu` provides the Python-facing CPU device facade that mirrors selected `torch.cuda` stream, event, synchronization, and device-query APIs for device-agnostic code. It also exposes cached CPU feature discovery and deprecated CPU AMP compatibility wrappers that delegate to the unified `torch.amp` implementation.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines CPU availability, device, stream, event, synchronization, and capability helpers |
| `amp/__init__.py` | Re-exports the CPU AMP compatibility classes `autocast` and `GradScaler` |
| `amp/autocast_mode.py` | Implements deprecated `torch.cpu.amp.autocast` as a CPU-specialized subclass of `torch.amp.autocast_mode.autocast` |
| `amp/grad_scaler.py` | Implements deprecated `torch.cpu.amp.GradScaler` as a CPU-specialized subclass of `torch.amp.GradScaler` |

## Public Interface

| Symbol | Description |
|---|---|
| `get_capabilities()` | Returns an immutable `MappingProxyType` over `torch._C._cpu._get_cpu_capability()` and caches the result with `lru_cache` |
| `_is_avx2_supported()` / `_is_avx512_supported()` / `_is_avx512_bf16_supported()` | Query x86 feature flags from `get_capabilities()` |
| `_is_vnni_supported()` / `_is_amx_tile_supported()` / `_is_amx_fp16_supported()` | Query VNNI and AMX flags from `get_capabilities()` |
| `_init_amx()` | Calls `torch._C._cpu._init_amx()` to initialize AMX support |
| `is_available()` / `is_initialized()` | Return `True` for the always-present CPU backend |
| `synchronize(device=None)` | Accepts the CUDA-like synchronization shape and performs no work because CPU execution is synchronous from this facade |
| `Stream` | No-op stream shim with `wait_stream`, `record_event`, and `wait_event` methods |
| `Event` | No-op event shim whose `query()` method returns `True` |
| `current_stream(device=None)` / `stream(stream)` / `StreamContext` | Read and temporarily replace the module-level `_current_stream` |
| `device_count()` / `set_device(device)` / `current_device()` | Report one CPU device, ignore device selection, and return `"cpu"` |
| `amp.autocast` | Deprecated CPU autocast wrapper that calls `torch.amp.autocast_mode.autocast("cpu", ...)` outside TorchScript |
| `amp.GradScaler` | Deprecated CPU gradient-scaler wrapper that calls `torch.amp.GradScaler("cpu", ...)` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch`, `torch.amp`, `torch._jit_internal`, `torch.types.Device`, and the public `torch.device` surface |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Uses `torch._C._cpu._get_cpu_capability()` and `torch._C._cpu._init_amx()` bindings from the C++ extension |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Reads `torch.cpu.get_capabilities()` and private feature helpers when choosing CPU vector ISA and codegen paths |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Lists `torch.cpu` helpers and CPU AMP contexts in trace rules and variable handling |

## Runtime Behaviour

`get_capabilities()` calls into `torch._C._cpu._get_cpu_capability()` once, wraps the returned dictionary in `MappingProxyType`, and serves all later feature queries from the `lru_cache`. The stream and event APIs intentionally perform no kernel scheduling: `Stream.wait_stream()`, `Stream.record_event()`, `Stream.wait_event()`, `Event.record()`, `Event.synchronize()`, and `Event.wait()` all return without side effects, while `Event.query()` returns `True`. `StreamContext.__enter__()` saves the module-level `_current_stream` and replaces it with the requested `Stream`; `__exit__()` restores the saved stream. The CPU AMP wrappers emit deprecation metadata and delegate non-TorchScript execution to the unified `torch.amp` classes with `"cpu"` as the device type.

## Performance Profile

The CPU capability path pays for the C++ capability probe only on the first `get_capabilities()` call; subsequent `_is_avx2_supported()`, `_is_amx_tile_supported()`, and related helpers perform dictionary lookups against the cached immutable mapping. Stream, event, synchronization, device-count, and device-selection calls perform constant-time Python work and do not synchronize with worker threads or native kernels. `StreamContext` updates one module global and has no locking, which keeps the shim cheap but makes it a device-agnostic compatibility layer rather than a scheduler. The deprecated AMP classes add one Python wrapper layer before entering `torch.amp`, so the autocast and scaling costs live in the shared AMP implementation rather than in `torch/cpu`.

## Design Rationale

`torch/cpu` gives generic code a CUDA-shaped API for CPU execution without pretending that CPU has multiple devices or asynchronous streams. The no-op stream and event classes let higher-level code share control-flow paths across CPU and accelerator backends while preserving the CPU backend's synchronous behavior. CPU feature queries centralize architecture detection behind `torch._C._cpu` and keep expensive probing out of hot loops through caching. The AMP subpackage preserves backwards compatibility for `torch.cpu.amp.*` names while steering new code to `torch.amp`.
