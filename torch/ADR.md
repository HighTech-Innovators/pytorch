# `torch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch` owns the public Python package for PyTorch. It assembles the C extension surface, Python tensor methods, lazy submodule loading, serialization helpers, and the top-level API names users import first.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Loads `torch._C`, populates the top-level namespace, handles global dependency loading, and lazy-loads selected heavy submodules |
| `_tensor.py` | Defines the Python `Tensor` subclass of `torch._C.TensorBase` and Python-level tensor behaviors such as `backward()` and `__repr__()` |
| `overrides.py` | Implements Python-side `__torch_function__` detection and dispatch helpers |
| `functional.py` | Exposes functional tensor APIs layered on top of the core package |
| `serialization.py` | Implements Python serialization entry points used by `torch.save` and `torch.load` |

## Public Interface

The package exports names collected in `__all__`, including `Tensor`, `SymInt`, `SymFloat`, `SymBool`, `save`, `load`, `no_grad`, `enable_grad`, `inference_mode`, `compile`, `rand`, `randn`, `matmul`, and `set_default_device`. `_tensor.py` exposes the Python `Tensor` type and methods such as `backward()`, while `overrides.py` exposes `handle_torch_function`, `has_torch_function`, `get_overridable_functions`, and `get_ignored_functions`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | `torch._C.TensorBase` and autograd bindings come from the compiled C extension built from `torch/csrc` |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `Tensor.backward()` delegates to `torch.autograd.backward(...)` |
| [torch/utils](torch/utils/ADR.md) | depends-on | `__init__.py` imports `_functionalize_sync`, `_import_dotted_name`, and internal path helpers from the utility layer |
| [torch/cuda](torch/cuda/ADR.md) | depends-on | `__getattr__` lazily imports backend namespaces such as `torch.cuda` when callers first access them |
| [torch/_inductor](torch/_inductor/ADR.md) | depends-on | `__getattr__` also lazily imports compiler-facing namespaces like `_inductor` and `_dynamo` |

## Runtime Behaviour

Importing `torch` runs `__init__.py`, decides whether to load `torch._C` with `RTLD_GLOBAL`, optionally calls `_load_global_deps()`, executes `from torch._C import *`, and then validates the presence of `_initExtension` to catch incorrect source-tree imports. The same file iterates over `dir(_C)` to extend `__all__`, binds `from torch import _C as _C`, and implements `__getattr__()` so names like `_dynamo`, `_inductor`, `_export`, and `onnx` import only on demand.

`_tensor.py` defines `class Tensor(torch._C.TensorBase)`, wraps many methods with `_handle_torch_function_and_wrap_type_error_to_not_implemented(...)`, and implements Python-side behaviors like subclass rebuilding, deep copy, `__repr__()`, and `backward()`. `overrides.py` supplies the Python fallback path for `__torch_function__`, so Python-defined APIs can respect tensor subclasses and override modes before entering the C++ dispatcher.

## Performance Profile

- **Allocation sites** - Importing `torch` allocates a large Python namespace and extends `__all__` by walking every public symbol from `_C`, while tensor subclass utilities in `_tensor.py` allocate Python state during deep copy, pickling, and subclass reconstruction.
- **Synchronization costs** - The top-level package does not add explicit runtime locks on operator execution, but GPU and autograd synchronization behavior becomes observable immediately once callers reach backend modules or call `Tensor.backward()`.
- **Data movement** - `torch.save` and `torch.load` route tensor data through `serialization.py`, and `_tensor.py` rebuild helpers move tensor state between Python objects and the underlying `TensorBase` handles.
- **Redundant or repeated work** - Lazy imports in `__getattr__()` avoid paying the import cost for `_dynamo`, `_inductor`, `_export`, and `onnx` until users touch those names, and `overrides.py` centralizes Python override checks so Python-defined APIs do not each duplicate `__torch_function__` dispatch code.

## Design Rationale

PyTorch keeps the public package thin over the compiled runtime so most operator semantics live in one C++ implementation while Python retains ergonomic APIs, subclass hooks, and import-time policy. The split between `__init__.py`, `_tensor.py`, and specialized subpackages makes the top-level namespace broad for users without forcing every heavy subsystem to import eagerly.
