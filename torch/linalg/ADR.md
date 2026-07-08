# `torch/linalg`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/linalg` owns the Python namespace for linear algebra operators. It binds high-level names such as `cholesky`, `inv`, and `cross` to built-in kernels and documents the numerical and synchronization semantics those kernels expose.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Connects `torch.linalg` symbols to `torch._C._linalg` builtins and attaches detailed operator documentation |

## Public Interface

Representative public entry points include `cross`, `cholesky`, `cholesky_ex`, `inv`, `inv_ex`, `solve`, `solve_ex`, `svd`, `eigh`, `qr`, `norm`, `matrix_norm`, and `vector_norm`. `LinAlgError` is re-exported from `torch._C` so callers can catch linear-algebra-specific failures with a namespace-local exception type.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | the namespace is a pure binding layer over `torch._C._linalg` builtins |
| [torch/_torch_docs](torch/_torch_docs/ADR.md) | depends-on | `__init__.py` formats operator documentation and shared notes in Python before exposing the builtins |

## Runtime Behaviour

`__init__.py` uses `_add_docstr` to bind Python names like `cholesky` and `inv` to native callables such as `_linalg.linalg_cholesky` and `_linalg.linalg_inv`. The module-level `common_notes` strings are embedded into many operator docstrings, so APIs such as `cholesky` and `inv` consistently explain when CUDA inputs synchronize with the CPU. Error-reporting variants such as `cholesky_ex` expose low-level `info` tensors directly, while checked variants such as `cholesky` build higher-level exceptions around those results. After argument binding, execution proceeds entirely in the underlying kernels.

## Performance Profile

The Python namespace adds almost no steady-state overhead because each symbol is a builtin function rather than a handwritten wrapper. The `_ex` variants explicitly avoid the slower error-string construction path and can skip synchronization unless callers request `check_errors=True`, which makes them the cheaper probe APIs. Batched linear algebra operations amortize kernel launches across many matrices, and the docstrings steer users toward faster formulations such as `solve` instead of `inv(A) @ B`. Memory reuse is delegated to kernel-level `out` parameters, so allocation behavior follows the backend implementation rather than Python glue code.

## Design Rationale

Linear algebra operations are grouped under `torch.linalg` so users get a coherent, NumPy-like namespace for matrix algorithms. The file stays declarative because backend kernels own numerical behavior, while Python owns discoverability, naming, and cross-operator documentation.
