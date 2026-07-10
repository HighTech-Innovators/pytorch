# `torch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch` is the top-level Python package and public API surface. Its `__init__.py` bootstraps the `torch._C` C extension, re-exports the tensor API, and makes all submodules (`torch.nn`, `torch.autograd`, `torch.distributed`, `torch.compile`, etc.) importable through a single namespace.

## Key Files

| File | Purpose |
|---|---|
| `torch/__init__.py` | Bootstrap: loads `torch._C` (`libtorch_python`), sets up tensor types, re-exports the full public API |
| `torch/serialization.py` | `torch.save` / `torch.load` (2252 lines): ZIP-format checkpoint serialization and `weights_only` deserialization |
| `torch/functional.py` | Functional operator wrappers exposed as `torch.*` (e.g., `torch.stack`, `torch.einsum`) |
| `torch/overrides.py` | `__torch_function__` protocol: `has_torch_function`, `handle_torch_function` |
| `torch/amp/` | Automatic mixed precision: `autocast` context manager |
| `torch/utils/` | `data.DataLoader`, `benchmark_utils`, `_pytree`, and other utilities |

## Public Interface

`torch.Tensor`, `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.load()`, `torch.save()`, `torch.compile()`, `torch.no_grad()`, `torch.inference_mode()`, `torch.device`, `torch.dtype`, `torch.amp.autocast`, and re-exports of every `torch.*` operator.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C` | depends-on | The C extension (`libtorch_python`) loaded in `__init__.py`; provides all C++ operator bindings |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Autograd is a subpackage; `torch.autograd.*` is accessed through this namespace |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | `torch.nn` is a subpackage |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | `torch.compile` dispatches through `_dynamo.optimize` |
| User code | depended-on-by | Every PyTorch user imports `torch` first |

## Runtime Behaviour

`import torch` triggers `__init__.py`, which calls `ctypes.CDLL` to load `libtorch_python` (or on some platforms relies on the Python import machinery for `.so` extension loading) and then calls `torch._C._initExtension()` / the module init function. This populates `torch._C` with tensor types, operator bindings, and extension module attributes. Submodules (`torch.nn`, `torch.autograd`) are loaded lazily on first attribute access or explicit import. `torch.serialization` provides `torch.save`/`torch.load`, which serialize tensors as ZIP files using the `StoragePtr`-based pickle protocol.

## Performance Profile

`import torch` cost is dominated by loading `libtorch_python` and running the `initModule()` chain in `torch/csrc/Module.cpp` — typically 0.5–2 seconds on first import due to dynamic linking and symbol resolution. Subsequent imports are free (module cached in `sys.modules`). `torch.load` with `weights_only=True` restricts the pickle deserializer to safe types; `weights_only=False` is faster but runs arbitrary unpickling. The `torch.*` operator surface itself adds no cost beyond the `torch._C` function-pointer dispatch already counted in `torch/csrc`.

## Design Rationale

Centralizing the public API in `torch/__init__.py` gives users one import point (`import torch`) while the internal implementation is split across a dozen subpackages. Re-exporting through the top level avoids exposing internal module paths in user-facing stack traces. The `__torch_function__` protocol in `overrides.py` lets third-party tensor-like objects intercept `torch.*` calls without modifying PyTorch source — the primary extensibility hook for NumPy arrays, sparse tensors, and custom tensor subclasses.
