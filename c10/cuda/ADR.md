# `c10/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/cuda` defines the CUDA runtime substrate below ATen kernels: device guards, stream objects, allocator interfaces, event helpers, error checks, and peer-access utilities. A CPU-only build does not execute these paths, but the source still defines the architecture that CUDA-enabled builds use for storage allocation, current-device state, stream routing, and launch diagnostics. Book Chapter 02 references this layer when it maps CUDA tensor storage to `CUDACachingAllocator` and device-aware `DataPtr` ownership.

## Key Files

| File | Purpose |
|---|---|
| `CUDACachingAllocator.h` | Public allocator namespace and `CUDAAllocator` interface for raw CUDA allocations, stream recording, snapshots, IPC, and mempools |
| `CUDAStream.h` | `CUDAStream` value wrapper around `c10::Stream` plus default, pooled, external, and current-stream accessors |
| `CUDAGuard.h` | CUDA-specialized RAII guards for current device and current stream state |
| `CUDAException.h` | `C10_CUDA_CHECK`, `C10_CUDA_KERNEL_LAUNCH_CHECK`, and device-side assertion launch helpers |
| `CUDAFunctions.h` | CUDA device-count, current-device, synchronization, and device-property helpers |
| `CUDAEvent.h` | CUDA event wrapper used for stream synchronization and timing semantics |

## Public Interface

The directory exposes the `c10::cuda` namespace and the `c10::cuda::CUDACachingAllocator` namespace. `CUDAStream` converts to `cudaStream_t`, exposes `device`, `device_index`, `id`, `query`, `synchronize`, `is_capturing`, and `priority`, and is acquired through `getStreamFromPool`, `getDefaultCUDAStream`, `getCurrentCUDAStream`, or `getStreamFromExternal`. `CUDAGuard`, `OptionalCUDAGuard`, and `CUDAStreamGuard` set and restore device or stream state with RAII. `CUDAAllocator` defines `raw_alloc`, `raw_alloc_with_stream`, `raw_delete`, `recordStream`, `snapshot`, mempool, IPC, and history hooks used by CUDA tensor storage.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `Device`, `DeviceIndex`, `Stream`, `DeviceGuard`, `DataPtr`, and allocator interfaces |
| [c10/util](c10/util/ADR.md) | depends-on | `Exception`, registry helpers, environment parsing, and range utilities |
| CUDA runtime and driver APIs | depends-on | `cuda_runtime_api.h`, `cuda.h`, streams, events, graph capture, and device properties |
| [aten/src/ATen/native/cuda](aten/src/ATen/native/cuda/ADR.md) | depended-on-by | CUDA kernels use streams, guards, math compatibility helpers, allocator state, and launch checks |
| [torch/cuda](torch/cuda/ADR.md) | depended-on-by | Python CUDA APIs expose allocator, stream, graph, and device state backed by this layer |

## Runtime Behaviour

`CUDAGuard` constructs an `InlineDeviceGuard<impl::CUDAGuardImpl>` and sets the current CUDA device from an index or `Device`, then restores the original state when the guard leaves scope. `CUDAStream` validates that wrapped streams carry `DeviceType::CUDA`, queries capture status with `cudaStreamIsCapturing`, queries priority with `cudaStreamGetPriority`, and converts to `cudaStream_t` for kernel launches and library calls. The stream pool has one default-stream pool plus low-priority and high-priority pools per device; low-priority streams are reused round-robin across 32 entries. `C10_CUDA_CHECK` sends CUDA error codes, source file, function, line, and device-assertion state to `c10_cuda_check_implementation`, while `C10_CUDA_KERNEL_LAUNCH_CHECK` checks `cudaGetLastError` immediately after a launch.

## Performance Profile

CUDA stream acquisition avoids runtime stream creation on the hot path because `getStreamFromPool` returns lazily created pooled streams rather than constructing a new `cudaStream_t` per request. `CUDACachingAllocator` hides expensive `cudaMalloc` and synchronization-heavy deallocation behind raw allocation, stream recording, expandable segment, and graph-capture-aware mempool hooks. The allocator exposes snapshots and trace history so memory tooling can inspect segments, traces, and host segments without changing the allocation interface. Device and stream guards compile to direct CUDA get/set operations for CUDA-linked code, which keeps current-device transitions cheap around short kernel launches.

## Design Rationale

Chapter 02 treats device placement as tensor metadata and places allocation behind `Allocator`/`DataPtr`; `c10/cuda` supplies the CUDA implementation of that contract. The design separates CUDA state from ATen math so generic tensor metadata remains backend-neutral while CUDA builds attach streams, events, and caching allocation only where needed. Pooled streams and cached blocks match CUDA workloads, where allocation and stream creation are far more expensive than ordinary C++ object construction. Error macros live next to CUDA runtime calls so every kernel launch and library call reports failures with source-local context.

