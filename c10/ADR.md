# `c10`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10` ("Caffe2 + ATen") is PyTorch's foundational C++ library. It owns the core tensor abstractions (`TensorImpl`, `Storage`), the memory-allocation interface, the device model, the scalar-type registry, and the `DispatchKey` machinery that routes every operator call. It carries zero Python dependency and compiles as a standalone shared library (`libc10.so`).

## Key Files

| File | Purpose |
|---|---|
| `c10/CMakeLists.txt` | Build definition for `libc10` — the base of the dependency stack |
| `c10/core/` | Tensor metadata, storage, dispatch keys, allocators, devices (see child ADR) |
| `c10/util/` | Low-level utilities: `intrusive_ptr`, `Exception`, `ThreadLocal`, `SmallVector` (see child ADR) |
| `c10/cuda/` | CUDA runtime types and caching allocator (see child ADR) |
| `c10/macros/` | Export macros (`C10_API`), platform/compiler shims |
| `c10/mobile/` | Binary-size-conscious mobile support code |

## Public Interface

`c10::TensorImpl`, `c10::Storage`, `c10::StorageImpl`, `c10::Allocator`, `c10::DataPtr`, `c10::Device`, `c10::DeviceType`, `c10::ScalarType`, `c10::DispatchKey`, `c10::DispatchKeySet`, `c10::SymInt`, `c10::intrusive_ptr<T>`, and the `TORCH_CHECK` / `TORCH_INTERNAL_ASSERT` error macros.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Tensor/storage/dispatch types defined in the `core` subtree |
| [c10/util](c10/util/ADR.md) | depends-on | Smart pointers, exceptions, containers used by every `c10` header |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | Optional CUDA runtime layer (not active in CPU-only deployment) |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen builds all operators on `c10` tensor and dispatch types |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | The Python bridge wraps `c10`/ATen tensors as Python objects |

## Runtime Behaviour

`c10` has no top-level entry point; it is a library linked into `libtorch`. Its runtime role is to provide always-present state: every tensor is a `c10::intrusive_ptr<TensorImpl>`, and `SetAllocator(DeviceType, Allocator*)` / `GetAllocator(DeviceType)` register and retrieve the per-device memory allocator at process startup. Dispatch-key registration also happens during static initialization, populating the per-operator tables that ATen later consults.

## Performance Profile

`c10` sits on the hottest paths in the framework: every operator call computes a `DispatchKeySet` (a `uint64_t` bitmask in `c10/core/DispatchKeySet.h`) and every tensor creation crosses the `Allocator::allocate()` boundary. Reference counting on `TensorImpl` uses atomic increments/decrements in `c10/util/intrusive_ptr.h` (`combined_refcount_`), so tensor churn in tight loops adds atomic-operation overhead. Because `c10` is binary-size conscious for mobile, it deliberately avoids heavyweight abstractions that would bloat these hot paths.

## Design Rationale

`c10` exists to give PyTorch a Python-free, ABI-stable core that can be deployed to mobile, C++ inference servers, and TorchScript without a Python runtime. The pImpl split between the thin `Tensor` handle and `TensorImpl` preserves ABI stability, and the `DispatchKey` table (rather than virtual methods on `TensorImpl`) lets thousands of operators and new backends register independently without editing the tensor core.
