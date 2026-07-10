# `c10/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/cuda` provides the CUDA-specific counterparts of the core `c10` abstractions: the CUDA caching allocator, CUDA stream and event wrappers, device guards, and CUDA error handling. It is the CUDA runtime layer that ATen's CUDA kernels build on. **In this project the deployment is CPU-only (built without CUDA), so this directory is documented for completeness but its code is not exercised at runtime.**

## Key Files

| File | Purpose |
|---|---|
| `c10/cuda/CUDACachingAllocator.cpp` | Caching allocator that pools freed CUDA blocks to avoid repeated `cudaMalloc` |
| `c10/cuda/CUDACachingAllocator.h` | Allocator interface, memory-stats, and pool configuration |
| `c10/cuda/CUDAStream.h` / `.cpp` | Wrapper over `cudaStream_t` with per-device stream pools |
| `c10/cuda/CUDAEvent.h` | RAII wrapper over `cudaEvent_t` for cross-stream synchronization |
| `c10/cuda/CUDAGuard.h` | `DeviceGuard` specialization that sets/restores the active CUDA device |
| `c10/cuda/CUDAFunctions.h` / `.cpp` | Thin wrappers over CUDA runtime queries (device count, current device) |
| `c10/cuda/CUDAException.h` / `.cpp` | `C10_CUDA_CHECK` and CUDA error translation to `c10::Error` |

## Public Interface

`c10::cuda::CUDACachingAllocator`, `CUDAStream`, `getCurrentCUDAStream()`, `getStreamFromPool()`, `CUDAEvent`, `CUDAGuard`, `OptionalCUDAGuard`, `device_count()`, `current_device()`, `set_device()`, `C10_CUDA_CHECK`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Implements `Allocator`/`DataPtr`, `DeviceGuardImplInterface`, `DeviceType::CUDA` |
| [c10/util](c10/util/ADR.md) | depends-on | `Exception`, `intrusive_ptr`, logging |
| CUDA runtime (`libcudart`) | depends-on | External CUDA toolkit; absent in CPU-only builds |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen CUDA kernels acquire streams/allocations here (inactive here) |

## Runtime Behaviour

When CUDA is present, the caching allocator in `CUDACachingAllocator.cpp` intercepts device allocations, first searching per-device free-block pools and only calling `cudaMalloc` on a cache miss; freed blocks return to the pool rather than to the driver. `CUDAStream` hands out streams from a per-device round-robin pool, and `CUDAGuard` sets the active device on construction and restores it on destruction. In this CPU-only build none of these code paths execute — no CUDA device is initialized and `DeviceType::CUDA` never appears in a live `DispatchKeySet`.

## Performance Profile

The central performance decision here is the caching allocator: `cudaMalloc`/`cudaFree` are synchronizing, high-latency calls, so pooling freed blocks converts most allocations into a fast in-process pool lookup and dramatically reduces allocation stalls in training loops. Stream pooling similarly amortizes stream-creation cost and enables kernel/copy overlap. Because this deployment is CPU-only, these mechanisms contribute no runtime cost or benefit here; the equivalent CPU allocation path lives in `c10/core/CPUAllocator.cpp`.

## Design Rationale

CUDA memory management is separated from `c10/core` so that the CPU-only core carries no CUDA dependency and `libc10.so` remains deployable without a CUDA toolkit. The caching-allocator design exists because naive per-tensor `cudaMalloc`/`cudaFree` would serialize the GPU pipeline; pooling is the standard mitigation. Keeping streams, events, and guards behind the same `c10` abstractions (allocator interface, `DeviceGuardImplInterface`) lets ATen kernels stay device-agnostic and lets a CPU-only build simply omit this subtree.
