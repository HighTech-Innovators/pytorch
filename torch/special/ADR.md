# `torch/special`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/special` owns the Python namespace for special mathematical functions. It binds names for gamma-family, error-function, Bessel, entropy, and polynomial operators to the native implementations and documents their mathematical contracts.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Connects `torch.special` symbols to `torch._C._special` builtins and exports the namespace through `__all__` |

## Public Interface

Representative public symbols include `entr`, `digamma`, `psi`, `gammaln`, `polygamma`, `erf`, `erfc`, `erfcx`, `erfinv`, `expit`, `i0`, `i1`, `ndtr`, `ndtri`, `xlog1py`, `xlogy`, `zeta`, `airy_ai`, `bessel_j0`, `bessel_y0`, and `spherical_bessel_j0`. `Tensor = torch.Tensor` is also exposed for the generated docs and callable signatures.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | every function in the namespace is a documented binding over a `torch._C._special` builtin |
| [torch/distributions](torch/distributions/ADR.md) | depended-on-by | distribution utilities and KL implementations call numerically stable special functions such as `xlogy` and `digamma` |

## Runtime Behaviour

The module binds names like `entr`, `digamma`, and `erf` to native builtins with `_add_docstr`, so imports populate the namespace and attach full math-oriented documentation in one pass. Aliases are expressed directly in Python: for example, `psi` is bound to `_special.special_psi` and documented as an alias for `torch.special.digamma`. Like `torch.fft` and `torch.linalg`, this file does not implement the numerical kernels in Python. After Python argument binding, all computation runs in the underlying builtin operators.

## Performance Profile

The namespace layer is effectively free compared with the native kernels, because each public symbol is a builtin rather than a Python loop or wrapper stack. Most functions are elementwise or small reductions, so their memory traffic follows input and output tensor sizes directly, and callers can often reuse storage via the optional `out` argument. Reusing native implementations also ensures special functions participate in device-specific vectorization and dispatch instead of paying Python overhead per element. The module's only repeated Python work is import-time docstring attachment.

## Design Rationale

Special functions are grouped under `torch.special` so mathematically specialized APIs stay discoverable and separate from the already large root namespace. A thin binding file is sufficient because the value of this package is naming, documentation, and organization, not Python-side computation.
