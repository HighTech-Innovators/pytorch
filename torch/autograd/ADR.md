# `torch/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/autograd` is the Python API surface for automatic differentiation. It exposes gradient computation, custom differentiable functions, gradient-mode context managers, and the Python-facing graph introspection API. The execution engine itself lives in `torch/csrc/autograd/engine.h`; this package provides the Python bindings and user-facing abstractions over it.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Package entry point; exports `backward`, `grad`, `Function`, `no_grad`, `enable_grad`, `inference_mode`, `gradcheck`, `gradgradcheck`; imports the C++ engine via `_engine_run_backward` from `graph.py` |
| `function.py` | `Function` — base class for user-defined differentiable operations; `FunctionCtx` stores saved tensors; `FunctionMeta` assigns unique IDs via `AUTOGRAD_FUNCTION_COUNTER` |
| `grad_mode.py` | `no_grad`, `enable_grad`, `inference_mode`, `set_grad_enabled` — thread-local RAII context managers backed by `torch._C._set_grad_enabled` and `c10::AutogradState` |
| `graph.py` | `Node` abstract base class (mirrors C++ `Node`); `GradientEdge`; `_engine_run_backward` — the Python entry point that calls the C++ backward engine; `saved_tensors_hooks`, `save_on_cpu` |
| `forward_ad.py` | Dual-tensor forward-mode AD; `dual_level()` context manager; `make_dual`, `unpack_dual` |
| `functional.py` | Functional differentiation: `jacobian`, `hessian`, `jvp`, `vhp`, `vjp` — all implemented by repeatedly calling `backward` or `forward_ad` |
| `gradcheck.py` | Numerical gradient checking via finite differences; `gradcheck`, `gradgradcheck` |
| `profiler.py` / `profiler_legacy.py` | Legacy `torch.autograd.profiler` — `profile`, `record_function`; superseded by `torch.profiler` but retained for backward compatibility |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.autograd.backward(tensors, grad_tensors)` | Triggers the C++ backward engine starting from `tensors`; calls `_engine_run_backward` |
| `torch.autograd.grad(outputs, inputs)` | Returns gradients without accumulating into `.grad` fields |
| `torch.autograd.Function` | User-defined differentiable op; subclass must implement `forward(ctx, ...)` and `backward(ctx, ...)` |
| `torch.autograd.no_grad()` | Context manager / decorator that disables gradient tracking for its scope |
| `torch.autograd.enable_grad()` | Re-enables gradient tracking inside a `no_grad` scope |
| `torch.autograd.inference_mode()` | Stronger than `no_grad`; tensors produced cannot be used as autograd inputs |
| `torch.autograd.graph.Node` | Abstract node in the autograd graph; `metadata`, `next_edges()` |
| `torch.autograd.graph.saved_tensors_hooks` | Context manager that overrides save/restore behaviour for saved tensors |
| `torch.autograd.gradcheck(func, inputs)` | Validates analytical gradient against numerical finite-difference approximation |
| `torch.autograd.forward_ad.make_dual` | Creates a dual tensor for forward-mode AD |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc](torch/csrc/ADR.md) | depends-on | `_engine_run_backward` calls `torch._C._EngineBase.run_backward`; `grad_mode.py` calls `torch._C._set_grad_enabled`; `c10::AutogradState` backs the context managers |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Autograd dispatch keys are registered in ATen; `at::Tensor::backward()` calls `_engine_run_backward` |
| [c10/core](c10/core/ADR.md) | depends-on | `AutogradState`, `InferenceMode`, `GradMode` thread-local state |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | `nn.Module` parameters have `requires_grad=True`; `nn.Module.backward_hooks` use autograd hooks |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimizers read `.grad` tensors populated by `backward()` |
| [torch/_functorch](torch/_functorch/ADR.md) | depended-on-by | AOT autograd and `grad` transform compose with this package |

## Runtime Behaviour

`backward(tensors, grad_tensors)` calls `graph._engine_run_backward`, which calls `torch._C._EngineBase.run_backward` — the C++ `Engine::execute` method in `torch/csrc/autograd/engine.cpp`. The C++ engine performs a BFS/topological traversal of the autograd graph starting from the provided `Node` edges, accumulating gradient tensors in `InputBuffer` objects and dispatching `Node::apply()` for each gradient function. Reentrant backwards (calling `backward` inside a custom `Function.backward`) are handled by the `MAX_DEPTH = 60` guard in `engine.h`; at that depth a new thread is spawned to avoid stack overflow. `no_grad` and `enable_grad` call `torch._C._set_grad_enabled(bool)`, which writes to a thread-local boolean in `c10::GradMode`. `inference_mode` uses `c10::InferenceMode` which additionally marks produced tensors as inference tensors.

## Performance Profile

- **Allocation sites**: every gradient-tracked tensor operation creates an autograd `Node` (heap-allocated via `std::make_shared<>` in the C++ engine) and calls `save_for_backward` to retain input tensors. Training loops create one node per operation per forward pass.
- **Synchronization costs**: the C++ engine's `ReadyQueue` uses a `std::mutex` and `std::condition_variable` to coordinate between the main thread and worker threads during backward. Multi-threaded backward (triggered by calling `backward` with `retain_graph=True` from multiple threads) requires careful lock ordering documented in `engine.cpp`.
- **Data movement**: `save_on_cpu` offloads saved tensors to CPU RAM during the forward pass and moves them back to GPU during backward, trading PCIe bandwidth for GPU memory headroom — an explicit activation-checkpointing alternative.
- **Redundant or repeated work**: `gradcheck` evaluates the function `2 * n_inputs + 1` times (central differences); it is intended only for testing and is extremely slow for large inputs.

## Design Rationale

`Function` (Python) and `Node` (C++) are kept as separate but parallel abstractions: Python-defined `Function` subclasses are called through the C++ engine via a `PyNode` wrapper, allowing user code to participate in the gradient graph without requiring C++ knowledge. `grad_mode` context managers are implemented as thin Python wrappers over C-level thread-local state in `c10::GradMode` so they add zero Python overhead to per-tensor dispatch. `inference_mode` is stricter than `no_grad` by design: it prevents produced tensors from accidentally entering the gradient graph in downstream code.
