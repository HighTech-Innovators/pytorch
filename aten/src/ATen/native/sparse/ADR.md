# `aten/src/ATen/native/sparse`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native/sparse` implements ATen operators for sparse COO and compressed sparse layouts. It owns sparse tensor construction, accessors, validation, math, sparse-dense BLAS, sparse binary intersections, and layout-specific stubs for CPU, CUDA, MPS, and other backends. Book Chapter 04 identifies sparse operators as a major native category and shows `native_functions.yaml` dispatch entries that route sparse keys into this directory.

## Key Files

| File | Purpose |
|---|---|
| `SparseTensor.cpp` | Sparse COO accessors, construction helpers, resize, coalesce state, indices/values access, and dense conversion entry points |
| `SparseTensorMath.cpp` | Sparse COO arithmetic, scalar operations, coalesce-sensitive math, and sparse add/mul/div/pow helpers |
| `SparseCsrTensor.cpp` | CSR, CSC, BSR, and BSC factory validation, compressed-index invariants, and compressed sparse tensor construction |
| `SparseBlas.cpp` | Sparse compressed BLAS-style operations such as `addmv`, triangular solve, and sampled addmm |
| `SparseBinaryOpIntersectionKernel.cpp` | CPU sparse-sparse intersection kernels for multiply and sparse mask projection |
| `SparseStubs.h` | Dispatch-stub declarations for sparse backend kernels |
| `ParamUtils.h` | Sparse parameter checking and normalization helpers |

## Public Interface

The public interface is the set of ATen operators whose YAML dispatch entries name sparse functions such as `add_sparse`, `add_out_sparse_cpu`, `_sparse_sum`, `_sparse_csr_sum_cpu`, `_sparse_softmax`, `_sparse_addmm`, and sparse compressed tensor factory functions. C++ functions in this directory operate on `SparseTensor`, regular `Tensor`, compressed sparse layouts, and dense operands. Accessor functions expose `sparse_dim`, `dense_dim`, `_nnz`, `_indices`, `_values`, `indices`, `values`, and coalesced state through generated ATen bindings.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Native schemas, generated op headers, TensorIterator, copy helpers, resize helpers, and dispatch stubs |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Tensor handle, generated operators, scalar/list types, and dispatcher integration |
| [c10/core](c10/core/ADR.md) | depends-on | Dispatch keys, layouts, devices, scalar types, and `TensorImpl` metadata |
| [c10/util](c10/util/ADR.md) | depends-on | `MaybeOwned`, `irange`, checked errors, and small metadata helpers |
| Sparse CUDA, MPS, XPU, and Eigen subdirectories | depends-on | Backend-specific sparse kernels and library adapters for sparse operations outside the top-level files |

## Runtime Behaviour

`SparseTensor.cpp` creates sparse tensors by choosing a `DispatchKey::Sparse*` key from the requested device type and constructing a `SparseTensorImpl` with the requested dtype. It exposes internal `_indices` and `_values` directly but requires `.coalesce()` before public `indices()` and `values()` return aliases, preserving the invariant that uncoalesced COO tensors do not expose duplicate-index values as canonical. `SparseTensorMath.cpp` operates on `_values()` for scalar multiplication, negation, and similar operations, copies `_indices()`, updates nnz through the sparse impl, and coalesces before floor division when uncoalesced duplicates would change results. `SparseCsrTensor.cpp` validates compressed layouts by checking strided index tensors, equal batch dimensions, block sizes, compressed/plain index dimensions, and layout-specific CSR, CSC, BSR, or BSC invariants.

## Performance Profile

Sparse kernels avoid dense work by operating on indices and values rather than materializing full tensors. Empty sparse matrices take fast paths in `SparseBlas.cpp`, including `addmv_out_sparse_compressed`, which returns zero or beta-scaled self when `mat._nnz() == 0`. `SparseBinaryOpIntersectionKernel.cpp` builds value-selection iterators over matching sparse indices and accumulates only the matched rhs values for each lhs nnz entry. `sparse_sampled_addmm_out_sparse_csr_cpu` transposes `mat2` to contiguous `[b, n, k]` layout before calling `sampled_addmm_sparse_csr_stub`, and it keeps result sparsity by resizing/copying the CSR pattern from `self`.

## Design Rationale

Chapter 04 presents sparse as a separate operator category because sparse tensors have different correctness and performance constraints from dense strided tensors. COO operations track coalesced state because duplicate indices change arithmetic semantics, and public accessors enforce coalescing before exposing canonical indices and values. Compressed sparse formats centralize invariant checks because a wrong compressed pointer array corrupts every downstream sparse BLAS operation. Sparse math stays in native code and backend subdirectories so YAML dispatch can route sparse CPU, CUDA, MPS, XPU, and meta keys without changing dense operator definitions.

