# `torch/sparse`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/sparse` owns the Python sparse-tensor namespace. It exposes sparse linear algebra and reduction operators for COO and compressed layouts and also hosts the prototype semi-structured sparse tensor subclass used for 2:4 sparsity.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Binds sparse builtins such as `addmm`, `mm`, and `sampled_addmm`, and exports sparse utility helpers |
| `semi_structured.py` | Defines `SparseSemiStructuredTensor` and backend-specific compressed sparse representations |
| `_semi_structured_ops.py` | Implements `__torch_dispatch__` handlers for semi-structured sparse values, transpose, matmul, addmm, linear, clone, and conversion |
| `_triton_ops.py` | Defines Triton-based sparse kernel helpers and layout validation routines |

## Public Interface

`addmm`, `mm`, `sampled_addmm`, `sum`, `softmax`, `log_softmax`, `check_sparse_tensor_invariants`, `as_sparse_gradcheck`, `SparseSemiStructuredTensor`, `SparseSemiStructuredTensorCUTLASS`, `SparseSemiStructuredTensorCUSPARSELT`, and `to_sparse_semi_structured` are the package-level interfaces. Semi-structured dispatch relies on methods and functions such as `SparseSemiStructuredTensor.__torch_dispatch__`, `_load_dispatch_table`, `semi_sparse_mm`, `semi_sparse_addmm`, `semi_sparse_linear`, and `semi_sparse_scaled_mm`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | `__init__.py` binds sparse builtins from `torch._C._sparse`, and semi-structured ops call ATen and custom sparse kernels |
| [torch/masked](torch/masked/ADR.md) | depended-on-by | `MaskedTensor` explicitly supports `torch.sparse_coo` and `torch.sparse_csr` layouts |
| [torch/multiprocessing](torch/multiprocessing/ADR.md) | depended-on-by | multiprocessing reducers rebuild sparse COO and compressed sparse tensors across process boundaries |

## Runtime Behaviour

`__init__.py` publishes builtin operators like `_sparse._sparse_addmm`, `_sparse._sparse_mm`, and `_sparse.sparse_sampled_addmm` under documented Python names. `SparseSemiStructuredTensor.__new__()` creates a wrapper subclass around compressed payload tensors such as `packed`, `meta`, `packed_t`, and `meta_t`, shows the prototype warning once, and loads a dispatch table that maps ATen overload packets to sparse-specific handlers. `_semi_structured_ops.py` implements those handlers: `semi_sparse_mm()` routes to backend-specific `_mm` kernels, `semi_sparse_t()` swaps stored transposed payloads without reconstructing dense data, and `semi_sparse_linear()` reuses `semi_sparse_addmm()` after flattening the leading dimensions. The package therefore supports both classic sparse layouts through builtins and semi-structured layouts through `__torch_dispatch__`.

## Performance Profile

Sparse COO and CSR operators reduce arithmetic and gradient traffic to nonzero entries, and `sum()` only propagates gradients at `nnz` locations. Semi-structured tensors store only half of the original 2:4 elements plus metadata, which cuts data movement and lets kernels such as `semi_sparse_mm()` and `semi_sparse_scaled_mm()` operate on compressed representations directly. The transpose path in `semi_sparse_t()` is cheap because it swaps precomputed compressed payloads instead of materializing a dense tensor. Sparse speedups depend on strict shape, dtype, and backend constraints, so validation logic in `semi_structured.py` and `_triton_ops.py` intentionally rejects unsupported cases rather than silently falling back to dense math.

## Design Rationale

PyTorch keeps sparse APIs in a dedicated namespace because sparse layouts, invariants, and operator coverage differ from dense tensors in important ways. Housing semi-structured tensor subclasses in the same package lets users discover emerging sparse formats next to the established COO and CSR operator set.
