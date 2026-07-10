# `torch/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/autograd` is the Python surface of automatic differentiation. It exposes `backward()`/`grad()`, the user-subclassable `Function` API, the gradient-mode context managers (`no_grad`, `enable_grad`, `inference_mode`), forward-mode AD, gradient checking, and the anomaly-detection tools. It is the thin Python layer that drives the C++ engine.

## Key Files

| File | Purpose |
|---|---|
| `torch/autograd/__init__.py` | Public `backward()`, `grad()`, `ProfilerActivity`; `_engine_run_backward` entry |
| `torch/autograd/function.py` | `Function` base class, `FunctionCtx`, `save_for_backward`, `apply` |
| `torch/autograd/grad_mode.py` | `no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode` |
| `torch/autograd/graph.py` | `_engine_run_backward` bridge to the C++ engine; `Node`/`saved_tensors` hooks |
| `torch/autograd/forward_ad.py` | Forward-mode (dual number) AD API |
| `torch/autograd/gradcheck.py` | Numerical gradient verification for tests |
| `torch/autograd/anomaly_mode.py` | `detect_anomaly` for NaN/backward-error diagnosis |
| `torch/autograd/functional.py` | `jacobian`, `hessian`, `vjp`, `jvp` |

## Public Interface

`torch.autograd.backward()`, `torch.autograd.grad()`, `torch.autograd.Function`, `Function.apply()`, `FunctionCtx.save_for_backward()`, `torch.no_grad`, `torch.enable_grad`, `torch.inference_mode`, `torch.set_grad_enabled`, `torch.autograd.gradcheck()`, `torch.autograd.detect_anomaly()`, `torch.autograd.functional.jacobian()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | `_engine_run_backward` calls `Engine::execute`; `PyNode` wraps `Function` |
| `torch._C` | depends-on | Grad-mode flags and engine bindings live in the C++ extension |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | Module forward calls build the graph this API differentiates |

## Runtime Behaviour

`loss.backward()` calls `torch/autograd/__init__.py::backward()`, which normalizes grad tensors and calls `_engine_run_backward()` in `graph.py`, crossing into the C++ `Engine`. A user `Function.apply(*args)` (in `function.py`) checks `is_grad_enabled()`, runs `forward(ctx, *args)`, records saved tensors through the C++ `SavedVariable` system, builds edges from input tensors, wraps the Python function in a C++ `PyNode`, and attaches `grad_fn` to the outputs. The context managers in `grad_mode.py` flip thread-local flags: `no_grad` disables recording, while `inference_mode` skips `AutogradMeta` allocation entirely.

## Performance Profile

This layer is thin, so its own cost is small, but the mode it selects has large runtime consequences: leaving autograd enabled on inference paths forces `TensorImpl` to allocate `AutogradMeta` and the engine to build no-op backward nodes, adding allocation and bookkeeping overhead per op. `inference_mode()` removes that entire cost and is the recommended inference fix. `save_for_backward` uses weak references where possible to avoid reference cycles that would delay tensor deallocation. Python-defined `Function.backward` bodies reacquire the GIL during backward, serializing that portion of the pass.

## Design Rationale

Keeping the differentiation *policy* (what to record, which mode is active) in Python while delegating *execution* to the C++ engine gives users a Pythonic, extensible API (`Function` subclasses, context managers) without paying Python overhead in the hot backward traversal. The explicit `save_for_backward`-vs-`ctx` attribute distinction separates tensor state (which must integrate with double-backward and anomaly detection) from plain Python data. `inference_mode` exists as a stronger `no_grad` precisely to let deployment code opt out of all autograd machinery — the primary lever in this CPU-only inference context.
