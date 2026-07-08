# `torch/_subclasses`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_subclasses` owns tensor-subclass infrastructure used by PyTorch compilers, most notably FakeTensor and FunctionalTensor. It models device and shape semantics without real compute, removes mutation through functionalization, and reconstructs meta or wrapper subclasses from real tensor metadata.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `FakeTensor`, `FakeTensorMode`, `DynamicOutputShapeException`, `UnsupportedFakeTensorException`, and `CrossRefFakeMode`. |
| `fake_tensor.py` | Implements `FakeTensor`, `FakeTensorMode`, fake dispatch caching, real-tensor propagation checks, and fake tensor conversion. |
| `fake_impls.py` | Registers Python fake-kernel fallbacks and per-op fake dispatch implementations. |
| `functional_tensor.py` | Implements `FunctionalTensor`, `FunctionalTensorMode`, and the Python, C++, and functorch functionalization APIs. |
| `meta_utils.py` | Implements `MetaConverter` and related helpers for rebuilding meta or wrapper subclasses from tensor descriptions. |

## Public Interface

The exported surface includes `FakeTensor`, `FakeTensorMode`, `UnsupportedFakeTensorException`, `DynamicOutputShapeException`, and `CrossRefFakeMode`. Other important symbols used across the compiler stack are `FunctionalTensor`, `FunctionalTensorMode`, `PythonFunctionalizeAPI`, `CppFunctionalizeAPI`, `FunctorchFunctionalizeAPI`, `MetaConverter`, `disable_fake_tensor_cache()`, and `unset_fake_temporarily()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_prims_common](torch/_prims_common/ADR.md) | depends-on | FakeTensor and functionalization use stride, dtype, and memory-format helpers such as `suggest_memory_format`, `elementwise_dtypes`, and contiguity checks. |
| [torch/_library](torch/_library/ADR.md) | depends-on | FakeTensor imports `FakeScriptObject`, custom-op profiling helpers, and fake-kernel registrations produced by `torch.library`. |
| [torch/_higher_order_ops](torch/_higher_order_ops/ADR.md) | depends-on | Functionalization calls `handle_effects()`, auto-functionalization helpers, and higher-order-op fake registrations during dispatch. |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo relies on FakeTensorMode, MetaConverter, and functionalization wrappers to trace Python programs without running real kernels. |

## Runtime Behaviour

`FakeTensor.__new__()` requires a meta tensor `elem`, a `FakeTensorMode`, and a fake device, then builds a wrapper subclass with `_make_subclass()` and stores `fake_mode`, `fake_device`, optional `real_tensor`, and debug trace state. `FakeTensor.__torch_dispatch__()` intercepts tensor-subclass dispatch, finds the owning fake mode from the arguments, refuses re-entry when a fake mode is already active, and then redispatches the call with that mode enabled.

`FakeTensorMode.__torch_dispatch__()` delegates to `dispatch()`, which handles direct metadata queries, optionally routes through `_cached_dispatch_impl()`, and otherwise falls back to `_dispatch_impl()` for the real fake-kernel lookup and execution path. The mode also tracks class-wide cache statistics in `cache_info()`, validates device initialization rules in `avoid_device_init`, and cross-checks fake outputs against real outputs in `_maybe_infer_fake()` when real tensor propagation is enabled.

`FunctionalTensorMode.__torch_dispatch__()` decomposes eligible ops, auto-functionalizes mutable custom ops when no native Functionalize kernel exists, threads effect tokens through `handle_effects()`, and finally toggles the C++ Functionalize dispatch key with `_ForceDispatchKeyGuard` so the existing functionalization kernels run. `MetaConverter` in `meta_utils.py` memoizes storages and tensors, reconstructs subclass structure through `__tensor_unflatten__()`, and can attach real storage snapshots when `copy_data=True`.

## Performance Profile

- **Allocation sites** - FakeTensor allocates wrapper subclasses around meta tensors, FunctionalTensor allocates wrapper subclasses around functionalization wrappers, and `MetaConverter` maintains memo dictionaries for every converted storage and tensor id.
- **Synchronization costs** - These paths avoid device synchronization by design, but fake dispatch still performs many metadata reads and may execute additional validation when real tensor propagation or mismatch tracing is enabled.
- **Data movement** - Fake execution usually avoids copying data entirely, while `MetaConverter` can clone real storage when `copy_data=True` and functionalization may issue `copy_()` or replacement updates when replaying mutations in a purely functional form.
- **Redundant or repeated work** - `FakeTensorMode` keeps a class-level dispatch cache and explicit bypass counters, which cut repeated fake-kernel work for stable schemas and shapes but intentionally fall back when higher-order operators or symbolic state make cache keys unsafe.

## Design Rationale

Compiler-facing execution needs tensor objects that preserve PyTorch semantics for devices, aliasing, shape symbols, and mutation without performing real computation. This directory concentrates that behavior in a small set of wrapper subclasses and modes, so Dynamo, export, and higher-order operators can share one coherent fake and functional execution model.
