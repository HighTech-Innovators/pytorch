# `torch/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/cuda` provides the Python-facing CUDA runtime surface for PyTorch tensors, streams, events, graphs, allocator controls, and device queries. It keeps CUDA importable on systems without an initialized driver by deferring real CUDA setup until code calls `_lazy_init()`, `cudart()`, allocator APIs, or device APIs.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines lazy initialization, device guards, stream context helpers, capability checks, and public imports such as `CUDAGraph`, `Stream`, and `Event` |
| `memory.py` | Wraps CUDA caching allocator controls, memory statistics, snapshots, `CUDAPluggableAllocator`, `MemPool`, and `use_mem_pool` |
| `graphs.py` | Exposes `CUDAGraph`, graph capture helpers, graph export hooks, and `make_graphed_callables` |
| `tunable.py` | Exposes TunableOp controls for enabling tuning, reading and writing tuning CSV files, and offline GEMM tuning |
| `streams.py` | Provides Python stream and event classes imported by `__init__.py` |

## Public Interface

| Symbol | Description |
|---|---|
| `is_available()` | Reports whether PyTorch was compiled with CUDA and whether the runtime can see CUDA devices |
| `init()` / `_lazy_init()` | Initializes CUDA state, runs queued lazy calls, and raises `DeferredCudaCallError` with the original call site when a queued operation fails |
| `device` / `device_of` / `set_device()` | Switches or selects the current CUDA device through `_exchange_device` and `_maybe_exchange_device` C bindings |
| `Stream`, `Event`, `stream()` | Controls CUDA stream and event execution contexts |
| `CUDAGraph`, `graph()`, `make_graphed_callables()` | Captures, instantiates, replays, and exports CUDA graphs |
| `memory_stats()`, `memory_snapshot()`, `empty_cache()` | Reports and controls the CUDA caching allocator |
| `CUDAPluggableAllocator`, `MemPool`, `use_mem_pool()` | Routes allocations through custom allocators and allocator pools |
| `torch.cuda.tunable.enable()` / `tuning_enable()` | Toggles TunableOp execution and tuning through `torch._C._cuda_tunableop_*` functions |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch`, `torch._C`, tensor types, and shared utilities such as `_LazySeedTracker` and `_dummy_type` |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Uses C extension entry points including `_cuda_init`, `_cuda_getDeviceCount`, `_cuda_cudaCachingAllocator_raw_alloc`, and `_CUDAGraph` |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | Relies on the C++ CUDA device, stream, event, and allocator infrastructure surfaced through `torch._C` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | CUDA tensor operators executed from Python use ATen CUDA kernels after this module selects devices and streams |
| [torch/profiler](torch/profiler/ADR.md) | depended-on-by | Profiling and CUDA graph annotations observe CUDA stream, event, and graph activity |

## Runtime Behaviour

`_lazy_init()` uses `_initialization_lock` and thread-local `_tls.is_initializing` to serialize CUDA initialization while allowing queued seed calls to run after `torch._C._cuda_init()` succeeds. `caching_allocator_alloc()` selects the current device and stream, converts a `torch.cuda.Stream` to its raw `cuda_stream` integer, and calls `_cuda_cudaCachingAllocator_raw_alloc` inside a `torch.cuda.device(device)` guard. `CUDAGraph.capture_begin()` and the `graph` context manager capture work on the current stream, while `CUDAGraph.instantiate()` and `replay()` delegate graph execution to the C++ `_CUDAGraph` base.

## Performance Profile

Lazy initialization removes CUDA driver startup cost from `import torch.cuda`, but the first real CUDA call pays for `_cuda_init()` and all queued lazy calls. The native caching allocator keeps freed blocks reserved, so `empty_cache()` releases only unoccupied cached memory and does not increase memory available to live PyTorch tensors. CUDA graphs reduce CPU launch overhead after capture by replaying an instantiated graph, and `tunable.py` can spend extra time benchmarking GEMM implementations so later calls use a recorded fast solution.

## Design Rationale

The module concentrates Python CUDA policy in one package and keeps low-level execution in C++ bindings, so Python code manages ergonomics while ATen and c10 manage device execution. Lazy initialization lets applications import CUDA APIs unconditionally, detect missing drivers through `is_available()`, and still produce precise errors when a queued CUDA action fails. Separate `memory.py`, `graphs.py`, and `tunable.py` files keep allocator, graph, and autotuning APIs independent while `__init__.py` re-exports the common public surface.
