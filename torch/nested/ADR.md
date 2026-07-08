# `torch/nested`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nested` owns the public nested-tensor API for ragged and variable-length data. It constructs nested tensors from dense or jagged inputs, binds core nested operators, and registers internal jagged-layout overrides for common tensor operations.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports `as_nested_tensor`, `nested_tensor`, `to_padded_tensor`, `nested_tensor_from_jagged`, `narrow`, and `masked_select` |
| `_internal/nested_tensor.py` | Defines `NestedTensor` helpers such as `jagged_from_list`, `nested_view_from_values_offsets`, and `nested_from_padded` |
| `_internal/ops.py` | Registers jagged-layout operator implementations such as `linear_default`, `_softmax_default`, and `to_copy_default` |

## Public Interface

`to_padded_tensor`, `as_nested_tensor`, `nested_tensor`, `nested_tensor_from_jagged`, `narrow`, and `masked_select` are the main public entry points. Internal but source-visible extension points include `NestedTensor`, `_rebuild_njt`, `jagged_from_list`, `nested_view_from_values_offsets`, `register_func`, `lookup_jagged`, `linear_default`, and `_softmax_default`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | `_internal/ops.py` implements nested overrides for module-style math such as `linear_default` and `linear_backward_default` |
| [torch/multiprocessing](torch/multiprocessing/ADR.md) | depended-on-by | nested tensors are added to the reducer path through `rebuild_nested_tensor` and `reduce_nested_tensor` |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | `to_padded_tensor` binds `torch._C._nested.nested_to_padded_tensor`, and `__init__.py` uses internal nested constructors such as `torch._nested_view_from_buffer` |

## Runtime Behaviour

`__init__.py` registers `_NestedTensor` and `_rebuild_njt` with `torch.serialization.add_safe_globals`, so nested tensor state can participate in safe weights-only loading. `as_nested_tensor()` accepts either an existing nested tensor, a dense tensor batch, or a list of tensors, then selects strided or jagged construction based on the requested `layout`. For strided layout it may reshape a dense tensor into a flat buffer and call `torch._nested_view_from_buffer`, while for jagged layout it either builds offsets from a dense batch or calls `jagged_from_list()` to produce the nested representation. `_internal/ops.py` then supplies layout-specific operator implementations such as `linear_default`, `clone_default`, `_softmax_default`, and `_log_softmax_default` for dispatch on jagged tensors.

## Performance Profile

Nested tensors avoid explicit padding for ragged data, which reduces wasted memory and arithmetic on invalid positions compared with dense padded batches. `as_nested_tensor()` can reuse underlying storage when the input is already contiguous and the requested dtype and device match, but `to_padded_tensor()` always copies because dense and nested layouts do not share the same storage organization. Jagged representations keep values and offsets in compact tensors, which improves data movement for variable-length workloads. The internal operator table avoids converting back to padded dense form for supported operations, so repeated ops such as `linear_default` and `_softmax_default` can stay in nested form.

## Design Rationale

PyTorch separates nested-tensor construction from operator registration so the public API stays small while layout-specific behavior evolves internally. Supporting both strided and jagged layouts in one package lets callers choose between compatibility and compact ragged storage without leaving the `torch.nested` namespace.
