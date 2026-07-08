# `torch/_prims`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_prims` defines PrimTorch primitive operators and the small execution utilities that trace ordinary `torch.*` programs into those primitives. It owns the dispatcher registrations for the `prims` namespace, plus the remapping mode that rewrites public torch APIs into reference implementations.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines the `prims` libraries, `TensorMeta()`, `_make_prim()`, and the large catalog of primitive operators. |
| `context.py` | Implements `TorchRefsMode`, which remaps `torch.*` calls and `OpOverload` invocations to `_refs` and decomposition functions. |
| `executor.py` | Provides `execute()` and `make_traced()` for tracing a callable to FX under `TorchRefsMode`. |
| `debug_prims.py` | Registers debugging-only primitive operators consumed at the end of `__init__.py`. |
| `rng_prims.py` | Registers RNG-specific primitive operators consumed at the end of `__init__.py`. |

## Public Interface

The public operator surface is the large `__all__` list in `__init__.py`, including primitive ops such as `add`, `mul`, `as_strided`, `broadcast_in_dim`, `reshape`, `where`, `copy_to`, `empty_strided`, `svd`, `fft_r2c`, `fft_c2c`, `fft_c2r`, `_make_token`, and `_sink_tokens`. Other important entry points are `TensorMeta()`, `TorchRefsMode`, `execute()`, and `make_traced()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_prims_common](torch/_prims_common/ADR.md) | depends-on | Primitive schemas, metadata checks, and type aliases use `TensorLikeType`, `RETURN_TYPE`, stride helpers, and `backwards_not_supported()`. |
| [torch/_refs](torch/_refs/ADR.md) | depends-on | `TorchRefsMode` maps `torch.*` APIs and many `OpOverload` objects to `_refs` implementations and decomposition-table entries. |
| [torch/_higher_order_ops](torch/_higher_order_ops/ADR.md) | depends-on | The prims namespace imports `new_token_tensor()` to model effect tokens for `_make_token` and `_sink_tokens`. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depends-on | `__init__.py` imports `FakeTensor` and `FakeTensorMode` so primitive meta and backend-select logic can behave correctly under fake execution. |

## Runtime Behaviour

`_make_prim()` defines a primitive schema in the `Library("prims", "DEF")`, installs CompositeExplicitAutograd, BackendSelect, Autograd, and Meta implementations, and returns the dispatcher operator object with cached `schema`, `return_type`, `prim_meta_impl`, and `prim_impl` attributes attached. `TensorMeta()` constructs metadata-only tensors from either an example tensor or explicit `shape`, `strides`, `dtype`, and `device`, and many primitive meta implementations use it to predict result layout without running a real kernel.

`TorchRefsMode.__torch_function__()` checks whether a call is already a primitive or a metadata passthrough, then remaps regular `torch.*`, `torch.Tensor` methods, `OpOverload`, and `OpOverloadPacket` calls to `_refs` or `torch._decomp.decomposition_table` entries. `make_traced()` in `executor.py` wraps a Python callable with `wrapper_and_args_for_make_fx()`, enters `TorchRefsMode`, traces the wrapped callable with `make_fx()`, and then executes the resulting `GraphModule` through `execute()`.

`register_rng_prims()` and `register_debug_prims()` run at the bottom of `__init__.py`, so the namespace includes auxiliary primitives that are not manually spelled out in the central operator catalog. That final import-time registration step means the prim library is ready as soon as `torch._prims` is imported.

## Performance Profile

- **Allocation sites** - Primitive registration allocates dispatcher libraries and per-op wrapper closures once, while eager primitive execution usually allocates only the output tensors created by the underlying ATen implementation or primitive meta function.
- **Synchronization costs** - Primitive wrappers do not add explicit synchronization, but every call still runs a Python-side metadata check before dispatch so unsupported broadcasting or promotion is rejected before the kernel executes.
- **Data movement** - Many primitives are thin adapters over existing ATen kernels, so data movement follows the lower-level kernel; the main Python overhead is argument normalization, metadata-only tensor construction, and the optional autograd error wrapper from `backwards_not_supported()`.
- **Redundant or repeated work** - `TorchRefsMode` caches both the torch-to-refs map and the set of primitive functions with `functools.cache`, while `make_traced()` intentionally retraces each call site because its output depends on the wrapped callable and current executor choice.

## Design Rationale

PrimTorch needs a small, stable vocabulary that compilers can decompose into and reason about mechanically. This directory keeps that vocabulary explicit in the dispatcher while pairing it with a Python remapping mode, so higher-level APIs can be lowered into prims without rewriting user programs into a separate source language.
