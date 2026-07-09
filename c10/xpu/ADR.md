# `c10/xpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/xpu` owns the low-level C++ runtime integration for PyTorch's XPU backend. It exposes device discovery, SYCL queue based streams, events, peer access, and the caching allocator that ATen and Python-visible XPU tensors use through `c10::DeviceType::XPU`.

## Key Files

| File | Purpose |
|---|---|
| `XPUFunctions.h` / `XPUFunctions.cpp` | Device-count, current-device, raw `sycl::device`, `sycl::context`, property, and pointer-to-device helpers |
| `XPUCachingAllocator.h` / `XPUCachingAllocator.cpp` | `XPUAllocator`, `NativeCachingAllocator`, `DeviceCachingAllocator`, graph memory pools, allocation history, and `c10::SetAllocator(c10::kXPU, ...)` registration |
| `XPUStream.h` / `XPUStream.cpp` | `XPUStream` wrapper over `c10::Stream` and `sycl::queue`, stream pools, external queue interop, and current stream state |
| `XPUEvent.h` | Movable `XPUEvent` wrapper around `sycl::event` with record, block, query, synchronize, and elapsed-time support |
| `XPUDeviceProp.h` | `DeviceProp` struct generated from SYCL device, platform, aspect, Intel extension, and experimental property macro lists |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::xpu::device_count()` / `device_count_ensure_non_zero()` | Return the lazily discovered XPU device count, warning or throwing when no device is available |
| `c10::xpu::current_device()` / `set_device()` / `exchange_device()` / `maybe_exchange_device()` | Manage the thread-local `DeviceIndex` stored in `curDeviceIndex` |
| `c10::xpu::get_raw_device(DeviceIndex)` / `get_device_context()` | Return the selected `sycl::device` and shared default `sycl::context` |
| `c10::xpu::get_device_properties(DeviceProp*, DeviceIndex)` | Fill `DeviceProp` fields such as `name`, `global_mem_size`, `sub_group_sizes`, `has_fp16`, and `architecture` |
| `c10::xpu::XPUStream` | Represents a SYCL queue and converts to `sycl::queue&`, `sycl::queue*`, and `c10::Stream` |
| `getStreamFromPool()` / `getStreamFromExternal()` / `getCurrentXPUStream()` / `setCurrentXPUStream()` | Allocate pooled streams, wrap external in-order queues, and read or update the current stream |
| `c10::xpu::XPUEvent` | Records SYCL events, makes streams wait via `ext_oneapi_submit_barrier`, and measures elapsed time from profiling timestamps |
| `c10::xpu::XPUCachingAllocator::get()` / `raw_alloc()` / `raw_delete()` / `recordStream()` | Expose the active XPU caching allocator and stream-aware lifetime tracking |
| `c10::xpu::MemPool` | Owns XPU graph memory-pool handles and coordinates `createOrIncrefPool`, `beginAllocateToPool`, and `releasePool` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses `Device`, `DeviceIndex`, `Stream`, `DataPtr`, `CachingDeviceAllocator`, `AllocatorConfig`, and `c10::SetAllocator` |
| [c10/util](c10/util/ADR.md) | depends-on | Uses `TORCH_CHECK`, `TORCH_WARN`, `c10::call_once`, `c10::irange`, and `ska::flat_hash_map` helpers |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen factory and kernel paths allocate XPU tensors through the registered `c10::kXPU` allocator and submit work to `XPUStream` queues |
| [torch](torch/ADR.md) | depended-on-by | Python `torch.xpu` APIs expose the device, stream, event, and allocator behavior implemented here |

## Runtime Behaviour

Device discovery runs lazily through `initDevicePoolCallOnce()`, enumerates Level Zero SYCL platforms, prefers dGPU platforms over iGPU platforms, and stores devices plus a default context in `gDevicePool`. Stream state also initializes lazily: `getStreamFromPool()` calls `initXPUStreamsOnce()` and `initDeviceStreamOnce(device)`, then returns one of 32 queues from the selected priority pool using an atomic round-robin counter. External queue interop stores the `sycl::queue*` directly in the `StreamId` after checking that the queue is non-null, in-order, in the PyTorch context, and on the requested device. The allocator installs `native_allocator` at static initialization time with `c10::SetAllocator(c10::kXPU, &native_allocator, 0)`, and `allocate()` uses `current_device()` plus `getCurrentXPUStream(device)` to associate each `DataPtr` with the active queue.

## Performance Profile

`XPUCachingAllocator.cpp` avoids raw SYCL allocation on every tensor allocation by keeping `small_blocks`, `large_blocks`, `active_blocks`, and stream event queues inside each `DeviceCachingAllocator`. Allocation sizes are rounded by `round_size()` and `get_allocation_size()`, and expandable segments reserve virtual memory with `reserve_virtual_mem` so physical pages can be mapped as needed. Stream allocation is cheap after initialization because each device owns fixed arrays of 32 queues per priority and `get_idx()` performs one atomic increment before constructing an `XPUStream` from a packed `StreamId`. Synchronization remains expensive: `XPUEvent::synchronize()` calls `event().wait_and_throw()`, `XPUStream::synchronize()` calls `queue().wait_and_throw()`, and `syncStreamsOnDevice()` walks every reserved queue in each priority pool because the file explicitly avoids device-wide wait while XPUGraph interop is unresolved.

## Design Rationale

The code keeps XPU support in `c10` so device, stream, event, and allocation primitives are available below ATen without depending on Python bindings. It uses SYCL queues as the concrete execution primitive while preserving PyTorch's backend-neutral `c10::Stream` and `c10::Allocator` interfaces. The device enumeration policy chooses one Level Zero platform and prefers dGPUs so the shared default `sycl::context` remains valid across all discovered XPU devices. The allocator mirrors the CUDA-style caching allocator contract but specializes the block model, stream events, peer access, and graph pools for SYCL and XPU graph capture.
