# `torch/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/cuda` is the Python CUDA device-management layer. It exposes availability checks, lazy initialization, device context managers, streams, events, memory allocator controls, CUDA graphs, random number state, NCCL helpers, NVTX ranges, AMP compatibility imports, and debugging utilities. It sits on the Python binding surface described in book Chapter 06 (`book/06-python-bindings.md`): Python functions call CUDA-enabled methods on `torch._C`, and CPU-only builds keep the module importable through dummy types and explicit runtime errors.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | CUDA module initialization, availability checks, queued calls, device APIs, properties, streams/events exports, and CPU-only fallbacks |
| `streams.py` | Python `Stream`, `ExternalStream`, and `Event` wrappers over `_CudaStreamBase` and `_CudaEventBase` |
| `memory.py` | CUDA caching allocator controls, raw allocation/free, memory stats, memory snapshots, memory pools, and allocator swapping |
| `graphs.py` | `CUDAGraph`, graph capture/replay helpers, graph memory pool handles, and cuda-bindings integration |
| `random.py` | CUDA RNG state get/set, seed, manual seed, and generator utilities |
| `comm.py` | Tensor scatter, gather, broadcast, and reduce helpers for multi-GPU communication |
| `nccl.py` | Python NCCL availability and collective wrappers backed by native CUDA/NCCL bindings |
| `nvtx.py` | NVTX range and marker helpers for external CUDA profiling tools |
| `_utils.py` | Device-index normalization and CUDA binding error utilities |
| `_memory_viz.py` | Formatting and visualization helpers for allocator memory snapshots |

## Public Interface

The package exports `is_available`, `is_initialized`, `device_count`, `current_device`, `set_device`, `device`, `stream`, `current_stream`, `default_stream`, `synchronize`, `get_device_properties`, `Stream`, `Event`, `ExternalStream`, `CUDAGraph`, `graph`, allocator APIs such as `memory_stats`, `memory_allocated`, `empty_cache`, `MemPool`, and CUDA graph utilities. It also exposes RNG APIs, AMP compatibility modules, NCCL helpers, profiler controls, and NVTX annotation functions. On builds without CUDA support, many symbols still exist but raise clear errors when a CUDA operation is requested.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/cuda](c10/cuda/ADR.md) | depends-on | Native CUDA device, stream, allocator, and runtime bindings ultimately use c10 CUDA abstractions |
| [c10/core](c10/core/ADR.md) | depends-on | Uses device objects, tensor devices, generators, and scalar/device metadata exposed through `torch._C` |
| [aten/src/ATen/native/cuda](aten/src/ATen/native/cuda/ADR.md) | depends-on | CUDA tensors and operations invoked from this module execute native CUDA kernels and allocator code |
| [torch/amp](torch/amp/ADR.md) | related | CUDA AMP compatibility modules re-export autocast and grad-scaler behavior for CUDA users |
| [torch/profiler](torch/profiler/ADR.md) | related | NVTX, CUDA profiling, memory snapshots, and graph annotations feed profiler and external observability workflows |

## Runtime Behaviour

Importing `torch.cuda` initializes Python state but does not initialize a CUDA context; `_lazy_init` drains queued calls only when a CUDA operation actually requires the runtime. `is_available` first checks whether the build has CUDA support and then uses either NVML-based discovery or the CUDA runtime device-count call, depending on environment configuration. `Stream` and `Event` wrap native base classes, record and wait through CUDA runtime semantics, and expose raw handles for interop protocols. The memory module routes raw allocations to the CUDA caching allocator with a device and stream, tracks stats through native calls, and exposes memory pools for scoped allocation behavior. CUDA graph helpers capture work on the current stream, preserve graph pool handles, and replay instantiated graphs through native `_CUDAGraph` objects.

## Performance Profile

Lazy initialization avoids CUDA driver startup cost for programs that import `torch` but never use CUDA. The caching allocator keeps freed blocks for reuse, tracks stream ownership, and reduces expensive `cudaMalloc`/`cudaFree` calls in training loops. Streams and events preserve asynchronous execution; synchronization APIs are explicit because unnecessary host waits destroy GPU overlap. CUDA graphs reduce repeated kernel-launch overhead by capturing a static sequence once and replaying it, while memory pools keep captured allocations stable across replays. Memory snapshots, NVML checks, and detailed stats add overhead only when users request them.

## Design Rationale

The module stays importable on CPU-only builds so library code can branch on `torch.cuda.is_available()` without crashing at import time. Python owns ergonomic device context managers and diagnostics, while `torch._C` owns actual CUDA runtime state, streams, events, graphs, and allocator operations. Streams, events, and graph wrappers expose raw handles because interoperation with CUDA libraries requires passing native runtime objects across package boundaries. The API keeps CUDA-specific features in `torch.cuda` while aligning names and semantics with other accelerator modules.
