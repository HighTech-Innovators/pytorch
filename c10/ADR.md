# `c10`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10` is the minimal-dependency C++ foundation library for PyTorch. It defines the core abstractions — tensor metadata, memory allocation interfaces, device representation, dispatch-key types, and C++ utilities — that both ATen and the torch Python layer depend on, without themselves depending on any other PyTorch component.

## Key Files

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Builds `c10` as an independent shared library; compiles `core/*.cpp`, `mobile/*.cpp`, `util/*.cpp`; sets no ATen or torch includes; enforces the dependency firewall |
| `core/TensorImpl.h` | Central tensor metadata class — shape, strides, dtype, device, storage pointer, dispatch key set, and Python object slot |
| `core/Allocator.h` | Abstract `Allocator` base class and `DataPtr` smart pointer; `GetAllocator`/`SetAllocator` registry used by all device allocators |
| `core/DispatchKey.h` / `core/DispatchKeySet.h` | Enumeration of all dispatch backends and functionality keys; 64-bit bitset carried by every `TensorImpl` |
| `core/Device.h` / `core/DeviceType.h` | `Device` struct (type + ordinal) and `DeviceType` enum covering CPU, CUDA, HIP, XLA, MPS, XPU, HPU, MTIA, Meta, and private-use slots |
| `util/intrusive_ptr.h` | `intrusive_ptr<T>` and `weak_intrusive_ptr<T>` — reference-counted smart pointers used for `TensorImpl`, `StorageImpl`, and all core heap objects |
| `util/ArrayRef.h` | Non-owning span over a contiguous array; used pervasively for shape and stride arguments throughout ATen |
| `util/Exception.h` | `TORCH_CHECK`, `TORCH_INTERNAL_ASSERT`, `Error`, `TypeError`, `ValueError` macros and classes |
| `mobile/CPUCachingAllocator.h` / `mobile/CPUProfilingAllocator.h` | Mobile-specific CPU allocators: arena-style caching allocator and profiling allocator that records allocation lifetimes |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::TensorImpl` | Core tensor metadata object; holds `SizesAndStrides`, `DataPtr`, `DispatchKeySet`, device, dtype, and `PyObjectSlot` |
| `c10::Storage` / `c10::StorageImpl` | Reference-counted raw memory handle with device annotation; shared across tensor views |
| `c10::Allocator` | Abstract allocator base class; `GetAllocator(DeviceType)` / `SetAllocator(DeviceType, Allocator*)` registry |
| `c10::Device` / `c10::DeviceType` | Device identity — type enum plus optional ordinal index |
| `c10::DispatchKey` / `c10::DispatchKeySet` | Backend and functionality key enumeration; 64-bit bitset for fast key-set operations |
| `c10::ScalarType` | Scalar dtype enumeration (`float`, `double`, `int64`, `bool`, `bfloat16`, etc.) |
| `c10::intrusive_ptr<T>` / `c10::weak_intrusive_ptr<T>` | Thread-safe reference-counted smart pointers |
| `c10::ArrayRef<T>` | Non-owning contiguous array view |
| `c10::Error` / `TORCH_CHECK` / `TORCH_INTERNAL_ASSERT` | Exception and assertion infrastructure |
| `c10::AutogradState` | Thread-local autograd mode flags (`GradMode`, `InferenceMode`) |
| `c10::CUDAStream` (via c10/cuda) | CUDA stream wrapper; 32-slot pool per device |
| `c10::CUDACachingAllocator` (via c10/cuda) | Block-pooled CUDA memory allocator implementing `c10::Allocator` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Defines the core tensor, storage, allocator, device, and dispatch-key abstractions that c10 exposes |
| [c10/util](c10/util/ADR.md) | depends-on | Provides intrusive pointers, ArrayRef, exception macros, and numeric types consumed by c10/core |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | CUDA-specific allocator and stream implementation that extends the core allocator interface |
| [c10/mobile](c10/mobile/ADR.md) | depends-on | Mobile CPU allocator and profiling allocator built on top of the core allocator interface |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen operator implementations, dispatch machinery, and native kernels all depend on c10 types |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | pybind11 bridge depends on `c10::TensorImpl`, `c10::Device`, `c10::ScalarType` for Python↔C++ conversion |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Autograd C++ engine reads `c10::AutogradState` thread-local flags and `TensorImpl` autograd metadata |

## Runtime Behaviour

At process startup, `c10`'s allocator registry is populated by static initialisers: the CPU allocator (`CPUAllocator.cpp`) registers via `SetAllocator(DeviceType::CPU, ...)`, and optional device extensions (CUDA, XPU) register their own allocators when loaded. `TensorImpl` objects are heap-allocated via `c10::make_intrusive<TensorImpl>()` and managed entirely by `intrusive_ptr`; their reference count is decremented to zero when the last Python or C++ reference is released, triggering destruction without a GC cycle.

`c10` itself carries no global mutable state beyond the allocator registry and the `AutogradState` thread-local. `DispatchKeySet` operations on `TensorImpl` are non-atomic word-sized reads and writes, which are safe only because the dispatch path holds no contested shared state — dispatch key mutation (e.g. `set_autograd_dispatch_key_`) happens only from the owning thread or under the GIL. The `CUDAStream` pool and caching allocator (in `c10/cuda`) introduce device-level locking for stream assignment and block recycling; these are the only synchronisation points within the library.

## Performance Profile

- **Allocation sites:** `c10::make_intrusive<TensorImpl>()` is called on every tensor creation path, placing one heap allocation per tensor on the critical path. `c10::DataPtr` wraps the raw buffer pointer; the buffer itself is allocated by the registered device allocator (CPU `posix_memalign`, CUDA caching allocator), so allocation cost is allocator-dependent. `c10/mobile/CPUCachingAllocator.h` introduces an arena-style caching allocator specifically to eliminate per-inference malloc overhead on mobile.
- **Synchronization costs:** The allocator registry (`GetAllocator`/`SetAllocator`) is a global array indexed by `DeviceType`; reads are unsynchronised after initialisation, which is safe only because writes happen only at startup. `c10/cuda/CUDACachingAllocator.cpp` acquires a per-device mutex on every `malloc` and `free` to manage free-block lists; this is the primary locking hotspot in the allocation path.
- **Data movement:** `c10` itself performs no data movement. `CopyBytes.cpp` in `c10/core` defines the `CopyBytes` dispatcher that selects memcpy vs. device-copy functions; actual copy is delegated to the registered backend.
- **Redundant work:** `SizesAndStrides` in `TensorImpl` uses a small-buffer optimisation (stores up to 5 dimensions inline) to avoid heap allocation for common tensor ranks, reducing the allocation load in shape-heavy operations.

## Design Rationale

The `CMakeLists.txt` comment makes the dependency policy explicit: "if you want to add ANY dependency to the c10 library, make sure you check with the core PyTorch developers as the dependency will be transitively passed on to all libraries dependent on PyTorch." This enforces the firewall that allows c10 to be used in constrained environments (mobile, server inference) without pulling in the full PyTorch build graph. The directory split into `core/`, `util/`, `cuda/`, and `mobile/` reflects a layered extension model: `util/` has no c10 dependencies, `core/` depends on `util/`, and device-specific directories (`cuda/`, `mobile/`) extend `core/` without coupling it to device runtimes.
