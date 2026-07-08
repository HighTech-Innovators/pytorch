# `torch/_numpy`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_numpy` implements a NumPy-like Python API on top of `torch.Tensor`. It wraps tensors in an `ndarray` facade, normalizes NumPy-style arguments, and forwards numerical work to existing torch kernels.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Builds the public `torch._numpy` namespace from dtype helpers, array constructors, ufuncs, linalg, and random helpers. |
| `_ndarray.py` | Defines the `ndarray` wrapper class, array constructors such as `array()` and `asarray()`, indexing helpers, and NumPy-style properties. |
| `_ufuncs.py` | Wraps unary and binary torch operations in NumPy-style argument normalization, casting, and `out=` handling. |
| `linalg.py` | Implements NumPy linalg functions by delegating to `torch.linalg` and translating error behavior. |
| `random.py` | Implements a subset of `numpy.random` using either torch RNG calls or an optional NumPy-backed stream. |

## Public Interface

The core array surface is `ndarray`, `array()`, `asarray()`, `ascontiguousarray()`, `from_dlpack()`, `can_cast()`, and `result_type()`. The module also exports many ufunc-style symbols from `_ufuncs.py`, including `matmul()`, `divmod()`, and the generated unary and binary operators, plus `linalg.*` functions like `solve()` and `svd()` and `random.*` functions like `rand()`, `randn()`, `choice()`, and `shuffle()`.

## Dependencies

This layer has no notable ADR-tracked Python dependencies beyond core `torch` and `torch.linalg` APIs. Its main collaboration surface is the public tensor API, which it wraps rather than extending another Python subsystem inside `src/torch`.

## Runtime Behaviour

The `ndarray` class stores a single wrapped tensor in `self.tensor`, then populates methods and dunder methods by iterating `methods`, `dunder`, and `ri_dunder` and binding them to functions from `_funcs` and `_ufuncs`. Property accessors like `shape`, `dtype`, `strides`, `real`, and `imag` forward directly to the wrapped tensor, while mutating methods such as `resize()` call `Tensor.resize_()` and explicitly zero-fill any newly grown contiguous region.

`_ufuncs.deco_binary_ufunc()` normalizes arguments, performs NEP 50 style scalar handling through `_dtypes_impl.nep50_to_tensors()`, applies explicit or inferred dtype casting, calls the underlying torch function, and finally routes results through `_ufunc_postprocess()` for `out=` broadcasting and casting. `linalg.py` wraps `torch.linalg` calls with `linalg_errors()` so `torch._C._LinAlgError` becomes `LinAlgError`, and `random.py` optionally swaps to `numpy.random` inside `deco_stream()` when `torch._dynamo.config.use_numpy_random_stream` is enabled.

## Performance Profile

- **Allocation sites** - The layer allocates lightweight `ndarray` wrappers freely, and helper paths such as `random_sample()` and `uniform()` allocate fresh tensors before wrapping or scalar-unwrapping them.
- **Synchronization costs** - The Python wrappers themselves do not synchronize devices, but conversions like `arg.tensor.numpy()` inside `random.deco_stream()` necessarily move execution onto host-visible arrays when the NumPy-backed stream path is enabled.
- **Data movement** - Most array views reuse the same underlying tensor, yet dtype conversions in `astype()`, `_ufunc_postprocess()`, and linalg promotion helpers allocate new tensors when NumPy semantics require a cast or broadcasted result.
- **Redundant or repeated work** - Ufunc wrappers intentionally repeat normalization, promotion, and `out=` checks around every call because NumPy compatibility is a surface-semantic guarantee, not a one-time registration property.

## Design Rationale

The directory preserves NumPy call shapes and error modes without introducing a new kernel layer by reusing torch operators everywhere possible. Keeping `ndarray` as a thin wrapper around `torch.Tensor` lets the compatibility surface stay broad while leaving performance-critical math in the existing tensor backend.
