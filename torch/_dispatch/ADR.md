# `torch/_dispatch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_dispatch` owns small Python-side helpers for manipulating dispatcher modes and validating functionalization behavior. It is a glue layer around dispatcher TLS toggles, FakeTensor cross-checks, and metadata comparison utilities.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the Python dispatcher context managers from `python.py`. |
| `python.py` | Defines mode toggles, functionalization suspension, metadata checks, and the cross-reference functionalization debugger. |

## Public Interface

The module exports `enable_python_dispatcher`, `no_python_dispatcher`, and `enable_pre_dispatch` as direct wrappers over `torch._C` dispatcher guards. Other entry points used by tracing and tests are `all_py_loaded_overloads()`, `suspend_functionalization()`, `enable_crossref_functionalize()`, `make_crossref_functionalize()`, `check_tensor_metadata_matches()`, and `check_metadata_matches()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_prims_common](torch/_prims_common/ADR.md) | depends-on | `check_tensor_metadata_matches()` delegates stride comparison to `torch._prims_common.check_significant_strides()`. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depends-on | `make_crossref_functionalize()` creates a `FakeTensorMode` and fakeifies tensors before replaying an operator. |
| [torch/_higher_order_ops](torch/_higher_order_ops/ADR.md) | depended-on-by | `BaseHOPFunction.backward()` and other higher-order operator paths call `suspend_functionalization()` to keep nested tracing stable. |

## Runtime Behaviour

`enable_python_dispatcher`, `no_python_dispatcher`, and `enable_pre_dispatch` are just aliases for `_EnablePythonDispatcher`, `_DisablePythonDispatcher`, and `_EnablePreDispatch`, so entering those contexts changes dispatcher TLS state in C++ without additional Python bookkeeping. `suspend_functionalization()` checks whether the `Functionalize` dispatch key is currently included, temporarily disables it with `torch._disable_functionalization()`, and restores the prior `reapply_views` setting on exit.

`make_crossref_functionalize()` builds a debug handler that fakeifies incoming tensors with `FakeTensorMode.from_tensor()`, runs the operator once under disabled modes plus suspended functionalization, then executes `op._op_dk(final_key, *args, **kwargs)` on the original arguments and compares pytree metadata with `check_metadata_matches()`. `enable_crossref_functionalize()` invalidates cached Functionalize dispatch for all Python-loaded overloads before and after the context so the cross-reference handler is reinstalled cleanly.

## Performance Profile

- **Allocation sites** - `make_crossref_functionalize()` allocates a fresh `FakeTensorMode`, fake tensor wrappers, detached argument pytrees, and formatted diagnostic strings for every checked operator call.
- **Synchronization costs** - The module does not issue device synchronizations, but every metadata cross-check reads sizes, dtypes, and strides from both fake and real outputs before returning control.
- **Data movement** - The debug handler avoids copying tensor data, but it does detach tensors, unwrap functional tensors with `torch._from_functional_tensor()`, and flatten nested pytrees twice per checked call.
- **Redundant or repeated work** - `all_py_loaded_overloads()` walks only lazily created `torch.ops` packets, and the file warns that `enable_crossref_functionalize()` is slow because it uncaches Functionalize dispatch on every loaded overload.

## Design Rationale

The directory stays intentionally small because most real dispatch logic lives in C++ or higher-level tracing stacks. Keeping these helpers in one file makes it easy for FakeTensor, higher-order operators, and functionalization debugging to share the same mode-toggling behavior.
