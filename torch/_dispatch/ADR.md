# `torch/_dispatch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_dispatch` provides Python-level helpers around PyTorch's dispatcher controls. Its `python.py` module exposes context managers for enabling or disabling Python dispatch paths and implements debugging utilities that cross-check Functionalize dispatch against fake-tensor execution.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Marks `torch._dispatch` as an importable package |
| `python.py` | Defines dispatcher context aliases, loaded-overload traversal, functionalization suspension, metadata comparison, and cross-reference functionalization debugging |

## Public Interface

| Symbol | Description |
|---|---|
| `enable_python_dispatcher` | Alias for `torch._C._EnablePythonDispatcher`; enables the Python dispatcher through a C++ context manager |
| `no_python_dispatcher` | Alias for `torch._C._DisablePythonDispatcher`; disables the Python dispatcher through a C++ context manager |
| `enable_pre_dispatch` | Alias for `torch._C._EnablePreDispatch`; enables pre-dispatch mode through a C++ context manager |
| `all_py_loaded_overloads()` | Iterates over lazily materialized `torch.ops` namespaces, packets, and overloads, yielding loaded `torch._ops.OpOverload` objects |
| `suspend_functionalization()` | Context manager that records Functionalize TLS state, disables functionalization when included, and restores it with the saved reapply-views flag |
| `check_tensor_metadata_matches(nv, rv, desc)` | Asserts size, dtype, and significant stride equality for two tensors |
| `check_metadata_matches(n, r, desc)` | Flattens pytree outputs and applies `check_tensor_metadata_matches()` to tensor leaves |
| `make_crossref_functionalize(op, final_key)` | Builds a handler that runs an operator under `FakeTensorMode`, dispatches the real operator with `op._op_dk(final_key, ...)`, and compares output metadata |
| `enable_crossref_functionalize()` | Debugging context that uncaches Functionalize dispatch handlers, enables the Python dispatcher, patches `CROSSREF_FUNCTIONALIZE`, and restores caches on exit |

Only `enable_python_dispatcher`, `no_python_dispatcher`, and `enable_pre_dispatch` appear in `__all__`; the other helpers support internal debugging and dispatcher integration.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch`, `torch._ops`, `torch.utils._python_dispatch`, `torch.utils._pytree`, tensor APIs, and `torch.ops` |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Uses `torch._C.DispatchKey`, `_EnablePythonDispatcher`, `_DisablePythonDispatcher`, `_EnablePreDispatch`, and dispatch TLS functions |
| [torch/_functorch](torch/_functorch/ADR.md) | depended-on-by | Uses `enable_python_dispatcher()` during AOTAutograd and graph capture |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | Uses `enable_python_dispatcher()` in shape propagation and proxy tensor paths |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Uses `enable_python_dispatcher()` while building variables, evaluating frames, and handling compiled autograd |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Uses `enable_python_dispatcher()` and `no_python_dispatcher()` in compilation, fake tensor, freezing, and pattern-matching paths |

## Runtime Behaviour

The three exported names in `__all__` directly bind C++ dispatcher context managers from `torch._C`, so `with enable_python_dispatcher():` changes dispatcher state through native TLS rather than through Python bookkeeping. `all_py_loaded_overloads()` walks the already-accessed `torch.ops` namespaces and yields allocated `OpOverload` objects; it does not force registration of every C++ operator. `suspend_functionalization()` reads whether `DispatchKey.Functionalize` is included in dispatch TLS and records `_functionalization_reapply_views_tls()` before calling `torch._disable_functionalization()`, then re-enables functionalization with the saved flag in `finally`. `enable_crossref_functionalize()` invalidates cached Functionalize handlers for all loaded overloads, enables Python dispatch, patches `torch._dispatch.python.CROSSREF_FUNCTIONALIZE`, and invalidates the same cache set again when leaving the context.

## Performance Profile

Normal use of `enable_python_dispatcher`, `no_python_dispatcher`, and `enable_pre_dispatch` costs one native context-manager transition plus the dispatcher behavior it enables inside the `with` block. `all_py_loaded_overloads()` scales with the number of `torch.ops` overloads already materialized in Python, and the source comment states that this list is useful for cache invalidation but is not complete. `enable_crossref_functionalize()` is deliberately slow: it traverses loaded overloads twice for `_uncache_dispatch(DispatchKey.Functionalize)`, and each handler built by `make_crossref_functionalize()` runs fake-tensor execution, real `op._op_dk()` dispatch, pytree flattening, and tensor metadata comparisons. The file keeps that cross-reference path behind an explicit debugging context instead of placing it on normal dispatcher execution.

## Design Rationale

`torch/_dispatch` keeps Python dispatcher toggles in a small module that compiler, export, FX, and subclass systems can import without depending on larger tracing subsystems. Aliasing the C++ context managers preserves a thin Python API while leaving authoritative dispatch state in `torch._C`. The cross-reference functionalization tools trade speed for strong diagnostics by comparing fake execution metadata with final-key dispatch results. The package-level `__init__.py` stays empty so importing `torch._dispatch` does not automatically touch dispatcher caches or load debugging helpers.
