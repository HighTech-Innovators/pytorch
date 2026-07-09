# `c10/core`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/core` owns PyTorch's foundational C++ abstractions: tensor metadata, memory ownership, device representation, and the dispatch-key type system. Every tensor operation in ATen and every Python-visible tensor passes through the types defined here.

## Key Files

| File | Purpose |
|---|---|
| `TensorImpl.h` | Central tensor metadata class — owns shape, strides, dtype, device, storage pointer, dispatch key set, and Python object slot |
| `Storage.h` / `StorageImpl.h` | Reference-counted raw memory handle with device annotation; wraps `DataPtr` |
| `Allocator.h` | Abstract memory allocator interface — `DataPtr`, `Allocator` base class, `GetAllocator`/`SetAllocator` registry |
| `CPUAllocator.cpp` / `CPUAllocator.h` | Default CPU allocator backed by `posix_memalign`; registered as device 0 allocator |
| `DispatchKey.h` / `DispatchKey.cpp` | Enumeration of all dispatch backends (`CPU`, `CUDA`, `HIP`, `XLA`, `MPS`, …) and functionality keys (`Dense`, `Quantized`, `Sparse`, …) |
| `DispatchKeySet.h` / `DispatchKeySet.cpp` | 64-bit bitset over `DispatchKey` values; carried by every `TensorImpl` |
| `Device.h` / `Device.cpp` | `Device` struct — type (`DeviceType`) plus ordinal index; `DeviceIndex` is `int8_t` |
| `DeviceType.h` / `DeviceType.cpp` | `DeviceType` enum covering CPU, CUDA, HIP, XLA, MPS, XPU, HPU, MTIA, PrivateUse1–3, Meta, and others |
| `ScalarType.h` | Scalar dtype enumeration (`float`, `double`, `int64`, `bool`, `bfloat16`, …) used as template dispatch tags |
| `AutogradState.h` / `AutogradState.cpp` | Thread-local autograd mode flags (`GradMode`, `InferenceMode`) |
| `CachingDeviceAllocator.h` | Interface for caching-style allocators (implemented per-device in ATen/CUDA) |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::TensorImpl` | Core tensor metadata; constructed by ATen factory ops; contains `sizes_and_strides_`, `storage_`, `key_set_`, `pyobj_slot_` |
| `c10::Storage` / `c10::StorageImpl` | Shared ownership of a raw memory buffer; `Storage` is the refcounted handle |
| `c10::DataPtr` | Unique pointer-with-deleter wrapping a raw memory allocation and its device |
| `c10::Allocator` | Abstract base; backends implement `allocate(size_t)` returning `DataPtr`; registered via `SetAllocator(DeviceType, Allocator*)` |
| `c10::GetAllocator(DeviceType)` | Returns the registered allocator for a device; used by ATen factory functions |
| `c10::DispatchKey` | Enum value identifying one dispatch backend or functionality |
| `c10::DispatchKeySet` | 64-bit bitset; queried by the dispatcher to select the next kernel |
| `c10::Device` | Struct holding `DeviceType` + `DeviceIndex`; stringified as `"cpu"`, `"cuda:1"`, etc. |
| `c10::ScalarType` | Enum for element dtype; `caffe2::TypeMeta` wraps it for runtime type metadata |
| `c10::AutogradState` | Thread-local state for grad-mode and inference-mode; read by `TensorImpl::set_requires_grad` |
| `c10::InferenceMode` | RAII guard that disables gradient tracking for a scope |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `c10/util` | depends-on | Exception macros, `intrusive_ptr`, `ArrayRef`, `DimVector`, `typeid` — all imported by TensorImpl.h |
| `c10/macros` | depends-on | `C10_API` export macros used throughout |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen wraps `TensorImpl`/`Storage` in `Tensor`; all ATen kernels read `DispatchKey` and `Device` from c10/core |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | pybind11 bindings convert `c10::Device`, `c10::ScalarType`, and `c10::TensorImpl` to Python objects |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Reads `AutogradState` and `TensorImpl::requires_grad_` |

## Runtime Behaviour

`TensorImpl` is constructed by ATen's factory functions (e.g., `at::empty`, `at::zeros`) and is reference-counted via `c10::intrusive_ptr`. The `Storage` field points to a separately refcounted `StorageImpl` that wraps a `DataPtr`; this separation enables views to share storage without copying data. The `DispatchKeySet` field is set at construction and updated whenever in-place operations change the tensor's properties (e.g., conjugate bit, grad tracking). `AutogradState` stores thread-local `GradMode` and `InferenceMode` state — both are RAII guards that push/pop a thread-local stack in `AutogradState.cpp`. There is no global lock on `TensorImpl` mutation; thread safety is the caller's responsibility and is documented as such in `TensorImpl.h`.

## Performance Profile

- **Allocation sites**: every tensor construction allocates a `TensorImpl` via `intrusive_ptr` (heap) and a `DataPtr` via the registered `Allocator`; for CPU tensors this goes through `CPUAllocator.cpp`'s `posix_memalign` path. Both allocations occur on every ATen factory call, making them hot in training loops.
- **Synchronization costs**: `c10::TensorImpl` uses `std::atomic<int>` for the intrusive reference count (in `c10/util/intrusive_ptr.h`). Reference count increments/decrements appear at every tensor copy and destruction. No mutex is held for normal tensor access; the `PyObjectSlot` acquires a per-interpreter lock only when the Python GIL handoff is needed.
- **Data movement**: `CopyBytes.cpp` provides the device-to-device copy primitive; it dispatches through a registry keyed on `(src_device, dst_device)` pairs. H2D and D2H copies route through this registry rather than being inlined.
- **Redundant work**: `DispatchKeySet` is recalculated on every `set_autocast_key_set` or `set_python_dispatch_key_set` call; these are rare but involve a 64-bit bitwise OR.

## Design Rationale

`c10/core` has zero external dependencies beyond system libraries, enforced by its `CMakeLists.txt`. This is a deliberate layering decision: `c10` must compile for mobile and embedded targets where ATen and its build infrastructure cannot be pulled in. The `Allocator` abstraction is kept as a pure virtual interface rather than a template so that different memory backends (CUDA caching allocator, mobile limited allocator, fake/meta allocator) can be registered at runtime without recompiling core. `DispatchKeySet` is a 64-bit integer rather than a `std::set` so that backend selection in the hot dispatcher path is a single 64-bit `tzcnt` instruction.
