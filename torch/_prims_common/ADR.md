# `torch/_prims_common`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_prims_common` centralizes shared type aliases, metadata rules, stride logic, and wrapper decorators used by PrimTorch and Python reference implementations. It is the common semantic layer that keeps `_prims`, `_refs`, FakeTensor, and related tracing code aligned on shape, layout, and type-promotion behavior.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines tensor and shape type aliases plus shared metadata, stride, contiguity, promotion, and device helpers. |
| `wrappers.py` | Implements reusable decorators and helpers for type promotion, `out=` handling, resizing, and safe output copies. |

## Public Interface

Frequently imported symbols include `ShapeType`, `StrideType`, `DimsType`, `TensorLikeType`, `TensorOrNumberLikeType`, `torch_function_passthrough`, `same_shape()`, `compare_tensor_meta()`, `check_significant_strides()`, `check_all_strides()`, `check_contiguous_sizes_strides()`, and `is_contiguous()`. Wrapper utilities include `elementwise_type_promotion_wrapper`, `out_wrapper`, `_maybe_convert_to_dtype()`, `_maybe_resize_out()`, `check_copy_devices()`, and `_safe_copy_out()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_subclasses](torch/_subclasses/ADR.md) | depends-on | `compare_tensor_meta()` raises `MetadataMismatchError` from FakeTensor when shapes, devices, strides, or conjugate state diverge. |
| [torch/_prims](torch/_prims/ADR.md) | depended-on-by | Primitive operator definitions import the type aliases, promotion helpers, and autograd wrappers declared here. |
| [torch/_refs](torch/_refs/ADR.md) | depended-on-by | Reference implementations import `TensorLikeType`, contiguity helpers, and decorators such as `out_wrapper()` and `elementwise_type_promotion_wrapper()`. |

## Runtime Behaviour

`same_shape()` and `compare_tensor_meta()` are the core metadata contracts: they compare lengths, symbolic sizes, dtypes, devices, strides, storage offsets, and conjugate or negative bits, and they use `guard_or_true()` or `guard_or_false()` when symbolic dimensions are involved. `check_significant_strides()` and `check_contiguous_sizes_strides()` deliberately ignore meaningless size-1 stride differences and recognize the `Max(1, size)` form that contiguous symbolic tensors can produce.

`elementwise_type_promotion_wrapper.__call__()` captures the wrapped function's signature once with `inspect.signature()`, then at runtime binds arguments with `_fast_bind()`, computes `compute_dtype` and `result_dtype` through `utils.elementwise_dtypes()`, casts promoted arguments, and converts outputs back to the requested dtype. `out_wrapper()` normalizes one or more `out` parameters, resizes mismatched zero-sized outputs with `_maybe_resize_out()`, and copies results with `_safe_copy_out()` after checking safe device and dtype casts.

## Performance Profile

- **Allocation sites** - The decorators allocate bound-argument dictionaries and sometimes temporary tuples for promoted sequences, but they avoid extra tensor allocation unless casting or output resizing is required.
- **Synchronization costs** - Metadata helpers inspect tensor properties only and never synchronize device work, which is why they are safe to call from tracing, FakeTensor, and decomposition code paths.
- **Data movement** - `_safe_copy_out()` is the main data-moving helper, and it only performs `copy_()` after device and dtype validation has already established that the target is reusable.
- **Redundant or repeated work** - `_fast_bind()` in `wrappers.py` avoids the heavier generic `inspect.Signature.bind()` path, and stride comparison routines skip non-significant dimensions so repeated metadata checks stay relatively cheap for broadcasted tensors.

## Design Rationale

PyTorch's compiler-facing Python layers need one shared definition of what counts as safe casting, matching metadata, or contiguous layout. Concentrating those rules here prevents `_prims`, `_refs`, and FakeTensor from drifting into subtly incompatible interpretations of the same tensor program.
