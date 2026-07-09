# `torch/sparse`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/sparse` provides the Python-facing sparse tensor namespace for sparse matrix operations, sparse reductions, invariant checking, sparse gradcheck support, semi-structured sparsity, and Triton-backed BSR helpers. It bridges public functions such as `torch.sparse.addmm` and `torch.sparse.softmax` to C++ `_sparse` bindings while keeping Python-only tensor subclass and tuning logic in this package.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines the public sparse namespace, docstrings, invariant controls, `sum()`, and `as_sparse_gradcheck()` |
| `_triton_ops_meta.py` | Stores and tunes precomputed Triton kernel parameters for sparse BSR operations |
| `_triton_ops.py` | Implements Python/Triton BSR scatter matrix multiplication and BSR dense addmm helpers |
| `semi_structured.py` | Defines `SparseSemiStructuredTensor`, CUTLASS and cuSPARSELt subclasses, and `to_sparse_semi_structured()` |
| `_semi_structured_conversions.py` | Converts dense tensors to and from CUTLASS semi-structured sparse representations |
| `_semi_structured_ops.py` | Implements `__torch_dispatch__` handlers for semi-structured values, indices, mm, addmm, linear, clone, view, and copy operations |

## Public Interface

`__init__.py` exports `addmm`, `mm`, `sampled_addmm`, `sum`, `softmax`, `log_softmax`, `spsolve`, `spdiags`, `check_sparse_tensor_invariants`, `as_sparse_gradcheck`, `SparseSemiStructuredTensor`, `SparseSemiStructuredTensorCUTLASS`, `SparseSemiStructuredTensorCUSPARSELT`, and `to_sparse_semi_structured`. `semi_structured.py` exposes wrapper-subclass behavior through `SparseSemiStructuredTensor.__new__()`, `__tensor_flatten__()`, `__tensor_unflatten__()`, `from_dense()`, `to_dense()`, and backend-specific `_mm()` implementations. `_triton_ops.py` provides internal acceleration helpers such as `scatter_mm()`, `bsr_scatter_mm_indices_data()`, `bsr_scatter_mm()`, `_int_bsr_dense_addmm()`, and `bsr_dense_addmm()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Public functions call `torch._sparse_sum`, `torch._C._sparse`, tensor constructors, tensor layouts, and wrapper-subclass APIs |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Sparse COO/CSR/CSC/BSR/BSC tensor storage, `_sparse` kernels, and low-level sparse operations are implemented in ATen |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `as_sparse_gradcheck()` extends autograd gradcheck and sparse matrix functions document backward behavior |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | `SparseSemiStructuredTensor.__new__()` calls `torch._dynamo.allow_in_graph()` for the tensor subclass |
| [torch/utils](torch/utils/ADR.md) | depends-on | `_triton_ops.py` uses `torch.utils._triton.has_triton` and Dynamo warning utilities for Triton availability |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Sparse Triton helpers and metadata feed compiler paths that lower sparse operations to generated kernels |

## Runtime Behaviour

Public sparse functions in `__init__.py` mostly attach docstrings to `_sparse` C++ bindings or call C++ entry points directly; for example, `sum()` delegates to `torch._sparse_sum()` with optional dimensions and dtype. `check_sparse_tensor_invariants` manages the process-wide C++ invariant flag through `torch._C._check_sparse_tensor_invariants()` and `torch._C._set_check_sparse_tensor_invariants()`, with context-manager and decorator forms. `to_sparse_semi_structured()` selects `SparseSemiStructuredTensorCUSPARSELT` by default unless `_FORCE_CUTLASS` is set, validates CUDA 2D dtype/shape constraints, and compresses dense inputs into packed tensors plus metadata. `_triton_ops.py` prepares BSR and dense inputs, computes or reuses indices data, handles empty and zero-alpha fast paths, and launches `scatter_mm()` for supported BSR formats.

## Performance Profile

Sparse public operations are intended to scale with specified entries or compressed blocks rather than dense tensor size, but layout choice controls constant factors and backend availability. `_triton_ops_meta.py` stores precomputed A100-oriented kernel parameters and falls back to reference parameters when the device or shape key is absent, trading portability for fast common sparse BSR configurations. `bsr_dense_addmm()` skips kernel work when `_nnz() == 0`, `alpha == 0`, or matrix dimensions are zero, and it broadcasts batch dimensions before launching scatter-style kernels. `SparseSemiStructuredTensor` stores packed values, metadata, and optional transposed packed forms so matrix multiplication and training paths avoid reconstructing dense tensors.

## Design Rationale

The package keeps Python-visible sparse ergonomics close to the root `torch.sparse` namespace while relying on ATen for fundamental sparse storage and kernels. Semi-structured sparsity uses tensor subclasses because compressed values and metadata need to behave like a tensor while intercepting operations through dispatch. Triton BSR support lives in Python so tuning metadata, fallback logic, and generated-kernel launch preparation can evolve independently from the stable C++ sparse layout implementations.
