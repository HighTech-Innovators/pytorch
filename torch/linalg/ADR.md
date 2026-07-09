# `torch/linalg`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/linalg` defines the Python `torch.linalg` namespace for linear algebra operations. The package binds names such as `cholesky`, `inv`, `eig`, `eigh`, `ldl_factor`, and `solve_ex` to C extension builtins from `torch._C._linalg` while attaching long-form public documentation and common synchronization notes.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Imports `_add_docstr`, `_linalg`, and `LinAlgError`, then binds documented public `torch.linalg` functions to `_linalg.linalg_*` builtins |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.linalg.cross` | Bound from `_linalg.linalg_cross`; computes batched 3D cross products with broadcasting |
| `torch.linalg.cholesky` / `cholesky_ex` | Bound from `_linalg.linalg_cholesky` and `_linalg.linalg_cholesky_ex`; compute Cholesky factors and optional LAPACK-style `info` |
| `torch.linalg.inv` / `inv_ex` | Bound from `_linalg.linalg_inv` and `_linalg.linalg_inv_ex`; compute matrix inverses with strict and `check_errors`-controlled variants |
| `torch.linalg.solve_ex` | Bound from `_linalg.linalg_solve_ex`; solves linear systems and returns `(result, info)` |
| `torch.linalg.det` / `slogdet` | Bound from `_linalg.linalg_det` and `_linalg.linalg_slogdet`; compute determinants and log-determinants |
| `torch.linalg.eig` / `eigvals` | Bound from `_linalg.linalg_eig` and `_linalg.linalg_eigvals`; compute general square-matrix eigendecompositions or eigenvalues |
| `torch.linalg.eigh` / `eigvalsh` | Bound from `_linalg.linalg_eigh` and `_linalg.linalg_eigvalsh`; compute Hermitian or symmetric eigen decompositions using the selected `UPLO` triangle |
| `torch.linalg.householder_product` | Bound from `_linalg.linalg_householder_product`; materializes products of Householder reflectors |
| `torch.linalg.ldl_factor` / `ldl_factor_ex` | Bound from `_linalg.linalg_ldl_factor` and `_linalg.linalg_ldl_factor_ex`; compute compact LDL factorizations and optional `info` |
| `torch.linalg.LinAlgError` | Alias for `_LinAlgError` imported from `torch._C` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch._C._add_docstr`, `torch._C._linalg`, and `_LinAlgError` from the top-level C extension |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Relies on Python C-extension bindings that expose `_linalg.linalg_*` builtins and `LinAlgError` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | The bound C++ implementations execute ATen linear algebra kernels, LAPACK-style routines, CUDA paths, and Tensor out variants |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Autograd formulas and checks consume the public linalg operators, including gradient stability constraints for eigenvector-producing functions |

## Runtime Behaviour

Importing `torch.linalg` executes a series of `_add_docstr(_linalg.linalg_*, r"""...""")` assignments, which attach Python documentation and publish the returned builtins under short namespace names. Runtime calls do not execute Python wrapper bodies in this file; they dispatch directly to the `_linalg` C extension functions assigned to names such as `cholesky`, `inv`, `eig`, and `ldl_factor_ex`. The documented `_ex` variants return LAPACK-style `info` tensors and defer error checking unless `check_errors=True`, while strict variants such as `cholesky` and `inv` raise `RuntimeError` for invalid decompositions or singular matrices. The docs state that many CUDA inputs synchronize with the CPU, and the `_ex` variants reduce synchronization by synchronizing only when `check_errors=True`.

## Performance Profile

The Python file has import-time cost proportional to the number and size of `_add_docstr` calls, but steady-state operator calls pay no extra Python-level algorithmic work from this module. Heavy computation happens in the `_linalg.linalg_*` builtins, which run ATen kernels for batched matrices and support float, double, cfloat, and cdouble dtypes across documented operations. The file explicitly steers users toward faster or more stable algorithms, for example recommending `torch.linalg.solve` instead of `torch.linalg.inv(A) @ B` and noting that `eigh` is faster than general `eig` for Hermitian or symmetric matrices. Synchronization dominates some GPU paths because the shared `common_notes` strings warn that CUDA inputs synchronize for strict variants and only conditionally synchronize for `check_errors`-controlled `_ex` variants.

## Design Rationale

The namespace keeps public API documentation close to the Python-visible binding names while leaving numerical kernels in ATen and C++ extension code. `_add_docstr` lets the module attach rich NumPy-style reference docs to builtins without adding Python forwarding layers. The strict and `_ex` pairs give users a clear choice between eager error reporting and lower-overhead access to backend `info` values. Central `common_notes` strings keep CUDA synchronization and experimental warnings consistent across related operations.
