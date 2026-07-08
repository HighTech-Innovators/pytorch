# `functorch/dim`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch/dim` implements first-class dimensions for tensors in Python. It owns `Dim` objects, `DimList` packs, wrapped tensor objects, dimension-aware indexing, reordering, and the dispatch glue that lets regular PyTorch operators act on those dimensions.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines `dims`, `dimlists`, `Dim`, `DimList`, `Tensor`, `stack`, `split`, `cat`, and the core `__torch_function__` dispatch path |
| `_wrap.py` | Wraps reduction-style torch operators so `Dim` objects can replace positional `dim` integers |
| `_getsetitem.py` | Implements first-class-dimension `__getitem__` and `__setitem__` handling, including `DimList` and dim-pack support |
| `_dim_entry.py` | Defines `DimEntry` and `_match_levels()` for aligning wrapped tensors by named or positional levels |
| `_order.py` | Implements `order()` for permuting and flattening first-class dimensions |
| `wrap_type.py` | Patches torch tensor methods and properties onto `_Tensor` through `__torch_function__` wrappers |
| `op_properties.py` | Enumerates pointwise operators that can use the faster pointwise dispatch path |

## Public Interface

The primary entry points are `dims()`, `dimlists()`, `Dim`, `DimList`, `Tensor`, `index()`, `stack()`, `split()`, `cat()`, and `order()`. Error handling uses `DimensionMismatchError` and `DimensionBindError`. Internal integration points include `_Tensor.__torch_function__()`, `_match_levels()`, `wrap_type()`, and `_wrap()`, which together attach dimension-aware behavior to regular tensor methods such as `sum`, `mean`, `max`, `argmax`, `softmax`, and the pointwise operators listed in `op_properties.py`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_C](torch/_C/ADR.md) | depends-on | Calls `torch._C._functorch._add_batch_dim` and `torch._C.TensorBase.__setitem__` from `Dim._get_batchtensor()`, `_Tensor.__torch_function__()`, and `setitem()` |
| [functorch/einops](functorch/einops/ADR.md) | depended-on-by | `functorch.einops.rearrange` imports `dims` and lowers patterns through `tensor[...]` plus `order()` |
| [torch/_tensor](torch/_tensor/ADR.md) | depended-on-by | `torch/_tensor.py` imports `functorch.dim.index` to expose first-class-dimension indexing from base tensors |

## Runtime Behaviour

`dims()` and `dimlists()` inspect caller bytecode with `_PyInstDecoder`, infer variable names from `STORE_*` or `UNPACK_SEQUENCE`, and create `Dim` or `DimList` objects that track binding state and sizes. `_Tensor.__torch_function__()` intercepts `torch.Tensor.__getitem__`, `torch.Tensor.__setitem__`, `torch.softmax`, wrapped reductions defined by `_def()`, and pointwise methods from `op_properties.py`, then rebuilds results through `TensorInfo`, `EnableAllLayers`, and `Tensor.from_positional()`. `_getsetitem.py` parses slices, tuples, `DimList` objects, and dim packs in `getsetitem()`, aligns RHS tensors with `_match_levels()`, and either forwards to the original tensor operation or reconstructs a first-class `Tensor`. `_order.py` reorders bound dimensions by matching `DimEntry` levels, optionally flattens groups of dimensions into one positional axis, and then renumbers positional dimensions from the right.

## Performance Profile

- **Allocation sites** - `Tensor.from_positional()` allocates wrapper objects around underlying tensors, while `Dim._get_range()` memoizes one `torch.arange()` tensor and `Dim._get_batchtensor()` memoizes one batched tensor per dimension. `Tensor.create_delayed()` also allocates placeholder wrappers for deferred pointwise multiplication results.
- **Synchronization costs** - The implementation uses no explicit locks; its overhead comes from repeated Python `__torch_function__` dispatch and `EnableAllLayers` context management around wrapped operators. `setitem()` and `getitem()` still delegate to `torch._C.TensorBase` once they finish dimension analysis.
- **Data movement** - `_match_levels()` uses `tensor.as_strided()` to realign dimensions without copying when the existing storage layout permits it. `order()` reshapes flattened groups after alignment, and `_add_batch_dims()` converts named dimensions into functorch batch dimensions with `_functorch._add_batch_dim`.
- **Redundant or repeated work** - `dims()` and `dimlists()` redo bytecode inspection on every constructor call so they can recover fresh variable names. `_Tensor.__torch_function__()` repeatedly rebuilds `TensorInfo` and level lists for operands, which adds Python bookkeeping cost in exchange for correct implicit batching semantics.

## Design Rationale

This subsystem stays in Python so it can prototype first-class dimensions without changing ATen operator schemas or kernels. The design reuses ordinary tensor operations, functorch batching machinery, and `__torch_function__` dispatch, which lets `Dim` semantics ride on top of existing PyTorch behavior instead of replacing it.
