# `c10/xpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/xpu` is the c10 runtime substrate for Intel XPU devices. It owns SYCL device discovery, streams, events, device guards, memory allocation, peer access, and device property queries that higher ATen and Python layers build on.

## Key Files

| File | Purpose |
|---|---|
| `XPUFunctions.cpp` | Enumerates SYCL devices, builds the shared context, tracks the current device, and populates `DeviceProp` |
| `XPUStream.cpp` | Manages per-device SYCL queue pools, `StreamId` packing, current streams, external queues, and device synchronization |
| `XPUEvent.h` | Wraps `sycl::event` with lazy creation, timing support, and stream wait helpers |
| `impl/XPUGuardImpl.h` | Implements `DeviceGuardImplInterface` for XPU device, stream, and event operations |
| `XPUCachingAllocator.cpp` | Implements the queue-aware caching allocator, expandable segments, pool capture support, and memory snapshots |
| `PeerToPeerAccess.cpp` | Caches `ext_oneapi_can_access_peer` results and enables allocator peer access on demand |
| `XPUDeviceProp.h` | Declares the `DeviceProp` struct and the SYCL property macros used to fill it |

## Public Interface

Other components call `device_count()`, `device_count_ensure_non_zero()`, `current_device()`, `set_device()`, `exchange_device()`, `get_raw_device()`, `get_device_context()`, `get_device_properties()`, and `get_device_idx_from_pointer()` from `XPUFunctions.h`. Stream users construct `XPUStream`, then use `getStreamFromPool()`, `getStreamFromExternal()`, `getCurrentXPUStream()`, `setCurrentXPUStream()`, and `syncStreamsOnDevice()`. Event and allocation users rely on `XPUEvent`, `impl::XPUGuardImpl`, `XPUCachingAllocator::recordStream()`, `XPUCachingAllocator::snapshot()`, `XPUCachingAllocator::createOrIncrefPool()`, `MemPool`, and `get_p2p_access()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Builds `Device`, `Stream`, `DeviceGuardImplInterface`, and `CachingDeviceAllocator` behavior on top of c10 core abstractions |
| [aten/src/ATen/xpu](aten/src/ATen/xpu/ADR.md) | depended-on-by | ATen XPU code consumes XPU device properties, streams, events, and allocator hooks from this layer |
| [torch/xpu](torch/xpu/ADR.md) | depended-on-by | Python `torch.xpu` APIs surface device counts, stream semantics, event timing, and allocator state defined here |

## Runtime Behaviour

`XPUFunctions.cpp` lazily enumerates Level Zero platforms in `enumDevices()`, prefers the first platform with dGPUs over iGPUs, creates one shared `sycl::context`, and stores the active device in thread-local `curDeviceIndex`. `XPUStream.cpp` lazily allocates 32 in-order queues per priority per device in `initDeviceStreamState()`, encodes queue identity into `StreamId`, rotates through each priority pool in `getStreamFromPool()`, and exposes external SYCL queues through `getStreamFromExternal()`. `impl/XPUGuardImpl` routes `DeviceGuard`, event recording, event timing, stream exchange, and device synchronization through `XPUStream`, `XPUEvent`, and `XPUCachingAllocator::recordStream()`. `XPUCachingAllocator.cpp` indexes cached `Block` objects by queue and size, records cross-stream uses in `recordStream()`, and can reserve and map virtual memory through `ExpandableSegment::map()` for expandable allocations.

## Performance Profile

- **Allocation sites** - `XPUCachingAllocator.cpp` allocates `Block` metadata, per-device block pools, and `ExpandableSegment` reservations, with 2 MB small segments and 20 MB large segments tracked by `ExpandableSegment`. `MemPool` wraps capture-specific pools and drops cached blocks in its destructor by calling `releasePool()` and `emptyCache()`.
- **Synchronization costs** - `syncStreamsOnDevice()` walks every reserved queue in every priority pool and calls `wait()` on it, so device sync cost scales with the fixed pool size. `XPUEvent::block()` and `XPUGuardImpl::record()` submit SYCL barriers, and allocator stream bookkeeping takes a `std::recursive_mutex` before mutating block state.
- **Data movement** - The allocator keeps blocks associated with the queue stored in `Block::queue`, which preserves reuse locality for allocations recorded on the same stream. External queues are keyed by raw `sycl::queue*`, so equivalent queues with different pointers do not share cached blocks.
- **Redundant or repeated work** - Lazy `initDevicePoolCallOnce()` and `initDeviceStreamOnce()` avoid repeated device enumeration and queue creation. `PeerToPeerAccess.cpp` memoizes the flattened `[num_devices x num_devices]` matrix in `p2pAccessEnabled_`, so hardware peer queries happen once per device pair.

## Design Rationale

PyTorch models XPU after its other accelerator backends: c10 owns the device, stream, guard, event, and allocator contracts, while ATen and Python layers stay above that line. The implementation maps those contracts onto SYCL queues and Level Zero constraints, which is why `XPUFunctions.cpp` centralizes platform selection and `XPUStream.cpp` encodes queue identity into a generic `StreamId`.
