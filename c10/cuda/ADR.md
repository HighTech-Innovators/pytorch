# `c10/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime Behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/cuda` provides the CUDA device abstraction layer: stream management, the CUDA caching allocator, device guards, and CUDA-specific exception handling. It implements the `c10::Allocator` interface for CUDA memory and registers CUDA streams with the c10 stream pool.

## Key Files

| File | Purpose |
|---|---|
| `CUDACachingAllocator.h` / `CUDACachingAllocator.cpp` | Block-pooled CUDA memory allocator; implements `c10::Allocator`; manages free-lists per stream and device |
| `CUDAStream.h` | `CUDAStream` wrapper over `cudaStream_t`; stream pool with 32 low-priority and 32 high-priority streams per device; round-robin allocation |
| `CUDAFunctions.h` / `CUDAFunctions.cpp` | `c10::cuda::device_count()`, `current_device()`, `set_device()` — thin wrappers over CUDA runtime API |
| `CUDAGuard.h` | RAII device guard that saves/restores the current CUDA device and stream on scope exit |
| `CUDAException.h` / `CUDAException.cpp` | `C10_CUDA_CHECK(status)` macro; translates `cudaError_t` to `c10::Error` |
| `CUDAAllocatorConfig.h` / `CUDAAllocatorConfig.cpp` | Runtime knobs for the caching allocator (max split size, garbage-collect threshold, CUDA graph pool) |
| `CUDAMallocAsyncAllocator.cpp` | Alternative allocator using `cudaMallocAsync` / stream-ordered memory |
| `CUDADeviceAssertionHost.h` / `CUDADeviceAssertionHost.cpp` | Host-side mechanism to catch GPU-side assertion failures and report them as Python exceptions |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::cuda::CUDAStream` | Stream wrapper; `getCurrentCUDAStream()`, `setCurrentCUDAStream()`, `getStreamFromPool()` |
| `c10::cuda::CUDAAllocatorConfig` | Static knob container; read at allocator initialization |
| `c10::cuda::CUDACachingAllocator::get()` | Returns the singleton `c10::Allocator*` for CUDA device memory |
| `c10::cuda::CUDACachingAllocator::emptyCache()` | Releases all cached free blocks back to CUDA |
| `c10::cuda::CUDAGuard` | RAII guard over `set_device` / `setCurrentCUDAStream` |
| `C10_CUDA_CHECK(expr)` | Evaluates CUDA runtime call; throws `c10::Error` with `cudaGetErrorString` on failure |
| `c10::cuda::device_count()` | Returns number of visible CUDA devices |
| `c10::cuda::current_device()` | Returns current CUDA device index |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `c10::Allocator`, `c10::DataPtr`, `c10::Device`, `c10::Stream` base types |
| [c10/util](c10/util/ADR.md) | depends-on | `c10::Error`, `TORCH_CHECK`, `intrusive_ptr` |
| CUDA runtime (`libcuda`, `libcudart`) | depends-on | `cudaMalloc`, `cudaFree`, `cudaStream_t`, `cudaDeviceSynchronize` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen CUDA kernels call `getCurrentCUDAStream()` and `CUDACachingAllocator::get()` |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Python CUDA API calls `device_count`, `set_device`, `emptyCache` |

## Runtime Behaviour

Stream pools are lazily initialized on first access per device. `getStreamFromPool()` returns streams in round-robin order from a fixed-size pool of 32 (no dynamic allocation after init). The caching allocator maintains per-device, per-stream free-lists of `Block` structs; `malloc` searches for a best-fit free block before calling `cudaMalloc`. When a block is freed (`raw_delete`), it is returned to the free-list rather than calling `cudaFree`, unless the pool exceeds the configured high-watermark. `CUDAGuard` saves the current device index via `current_device()` in its constructor and restores it in its destructor; it is stack-allocated and adds no heap overhead.

## Performance Profile

- **Allocation sites**: `CUDACachingAllocator::malloc` is called on every GPU tensor construction. The fast path — finding a free block in the pool — holds a per-device `std::mutex` for the duration of the search. Large allocations that miss the pool call `cudaMalloc`, which synchronizes the GPU and is expensive (~100 µs).
- **Synchronization costs**: The per-device mutex in `CUDACachingAllocator` is the primary contention point when multiple CPU threads allocate GPU tensors concurrently. `C10_CUDA_CHECK` introduces implicit `cudaPeekAtLastError` calls after every CUDA launch in debug builds.
- **Data movement**: H2D and D2H transfers are initiated by ATen's `copy_` kernel; `c10/cuda` provides the stream on which the transfer runs via `getCurrentCUDAStream()`.
- **Redundant or repeated work**: Caching the stream handle in `CUDAStream` avoids repeated `cudaStreamQuery` calls. Allocator fragmentation under mixed-size workloads can force `emptyCache()` followed by new `cudaMalloc` calls.

## Design Rationale

The caching allocator exists because `cudaMalloc` is synchronous and far too slow to call on every tensor operation. The block-pooling strategy (originally introduced in PyTorch 0.4) trades fragmentation for latency. The stream pool is fixed-size (32 per priority level) rather than dynamically growing because `cudaStreamCreate` is also synchronous; the pool amortizes that cost across all tensor operations on a device. `CUDAGuard` uses RAII rather than an explicit save/restore API to guarantee correctness in the presence of exceptions.
