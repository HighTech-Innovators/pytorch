# `torch/nested`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nested` provides PyTorch's public nested tensor API for representing batches with ragged dimensions. The package exposes construction and conversion helpers in `torch/nested/__init__.py` and implements the jagged tensor subclass, dispatch table, and scaled-dot-product-attention support in `torch/nested/_internal`.

## Key Files

| File | Purpose |
|---|---|
| `torch/nested/__init__.py` | Public namespace for `as_nested_tensor`, `nested_tensor`, `nested_tensor_from_jagged`, `narrow`, `masked_select`, and `to_padded_tensor`. |
| `torch/nested/_internal/nested_tensor.py` | Defines the `NestedTensor` wrapper subclass, jagged metadata layout, serialization hooks, `__torch_dispatch__`, and constructors such as `jagged_from_list`. |
| `torch/nested/_internal/ops.py` | Registers jagged implementations in `JAGGED_OPS_TABLE` and routes ATen operations through `lookup_jagged`. |
| `torch/nested/_internal/sdpa.py` | Selects Flash, efficient, math, or cuDNN scaled-dot-product-attention backends for jagged nested tensors. |

## Public Interface

| Symbol | Description |
|---|---|
| `as_nested_tensor(ts, dtype=None, device=None, layout=None)` | Builds a nested tensor while preserving autograd history; supports `torch.strided` and `torch.jagged` layouts. |
| `nested_tensor(tensor_list, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False)` | Builds a leaf nested tensor from a list of tensors or array-like values. |
| `nested_tensor_from_jagged(values, offsets=None, lengths=None, jagged_dim=None, min_seqlen=None, max_seqlen=None)` | Creates a `torch.jagged` nested tensor view over a packed values buffer and offsets or lengths metadata. |
| `narrow(tensor, dim, start, length, layout=torch.strided)` | Creates a nested tensor from a strided tensor; the jagged path supports `dim=1` and tensor starts or lengths. |
| `masked_select(tensor, mask)` | Builds a jagged nested tensor by preserving selected values and deriving offsets from the expanded mask. |
| `to_padded_tensor(input, padding, output_size=None, out=None)` | Python binding to `_nested.nested_to_padded_tensor`; always copies nested data into a padded dense tensor. |
| `NestedTensor` | Internal `torch.Tensor` wrapper subclass stored in `torch/nested/_internal/nested_tensor.py`; public constructors return instances for jagged layout. |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses `torch.Tensor`, `torch._nested_tensor_from_tensor_list`, `torch._nested_view_from_buffer`, `torch.serialization.add_safe_globals`, `torch.jagged`, tensor factories, tensor views, and `torch.ops`. |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Imports `_add_docstr` and `_nested` from `torch._C`, uses `DispatchKey.NestedTensor`, `DispatchKey.AutogradNestedTensor`, and dispatcher queries in `NestedTensor.__new__` and `__torch_dispatch__`. |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Registers handlers for ATen operators such as `aten.linear`, `aten.matmul`, `aten._softmax`, `aten.sum`, `aten._nested_get_offsets`, and scaled-dot-product-attention kernels. |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Uses `torch.autograd.Function` classes `ViewBufferFromNested` and `ViewNestedFromBuffer`, tracks `requires_grad`, and returns autograd-aware nested views. |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Marks ragged and sequence-length tensors dynamic with `_dynamo_propagated_dynamic_indices` and `torch._dynamo.mark_dynamic`. |
| [torch/fx](torch/fx/ADR.md) | depends-on | Rejects `nested_tensor_from_jagged` under `fx.symbolic_trace` and uses FX schema normalization in jagged op wrappers. |

## Runtime Behaviour

`torch/nested/__init__.py` routes strided layout creation to C++ `_nested` builtins and routes jagged layout creation to `torch.nested._internal.nested_tensor` helpers such as `jagged_from_list`, `jagged_from_tensor_and_lengths`, and `nested_view_from_values_offsets_lengths`. A jagged `NestedTensor` stores `_values`, `_offsets`, optional `_lengths`, `_ragged_idx`, `_size`, `_strides`, and `_metadata_cache`, then intercepts operations through `__torch_dispatch__` and `__torch_function__`. `lookup_jagged` checks `JAGGED_OPS_TABLE`, applies pointwise fallbacks, validates schemas with `check_schema`, and dispatches many operations to `_values` while preserving offsets and metadata. `jagged_scaled_dot_product_attention` validates nested QKV inputs, applies autocast manually, selects Flash, efficient, cuDNN, or math backend, and returns a jagged output with the query offsets and sequence-length cache.

## Performance Profile

Jagged nested tensors avoid padding by storing batch items in one packed `_values` buffer plus `_offsets` and optional `_lengths`; operations such as pointwise functions, `linear`, softmax, dropout, reductions, `matmul`, and `bmm` reuse dense kernels on `_values` when the ragged metadata permits it. Metadata operations can force synchronization: `_get_max_seqlen`, `_get_min_seqlen`, `_jagged_numel`, and SDPA preprocessing call reductions or `.item()` paths unless cached `min_seqlen` and `max_seqlen` tensors are provided. `to_padded_tensor` always copies because the nested and dense memory layouts differ, and the math SDPA fallback converts jagged tensors to strided nested tensors and back. The optimized SDPA path uses offsets, cumulative sequence lengths, and packed buffers to call `_flash_attention_forward`, `_efficient_attention_forward`, or `_cudnn_attention_forward` without padding each sequence to the batch maximum.

## Design Rationale

The public module keeps construction APIs small while the internal package owns the tensor subclass and operator surface. The jagged layout models one ragged dimension with explicit offsets so PyTorch can represent transformer sequence batches and key-value cache windows without dense padding. The dispatch table design lets the package add nested-specific behavior for individual ATen ops while falling back to dense `_values` operations for safe pointwise cases. The SDPA implementation selects backends in Python because it must combine nested metadata checks, CUDA backend capability checks, and layout conversion decisions before calling the low-level attention kernels.
