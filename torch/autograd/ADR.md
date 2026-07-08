# `torch/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/autograd` is the Python-facing automatic differentiation API layered over the C++ autograd engine described in book Chapter 05. It exposes `backward`, `grad`, custom `Function`, gradient-mode context managers, finite-difference validation, graph hooks, saved-tensor policies, forward-mode AD helpers, and functional Jacobian/Hessian utilities. The directory does not implement tensor kernels; it validates Python arguments, manages user-facing context, and delegates graph execution to `torch/csrc/autograd`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public `torch.autograd` namespace, `_make_grads`, `backward`, `grad`, and exports for grad mode, Function, gradcheck, and graph helpers |
| `function.py` | Defines `FunctionCtx`, `FunctionMeta`, `Function.apply`, `backward`/`vjp`, `jvp`, `vmap`, saved tensor APIs, and custom autograd Function plumbing |
| `grad_mode.py` | Defines `no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode`, and related thread-local gradient-state context managers |
| `functional.py` | Implements `vjp`, `jvp`, `jacobian`, `hessian`, `hvp`, and `vhp` by composing `grad`, forward AD, and vmap |
| `gradcheck.py` | Implements `gradcheck` and `gradgradcheck` by comparing analytical gradients with finite differences, including complex, sparse, batched, and forward-AD cases |
| `graph.py` | Exposes graph `Node`, `GradientEdge`, hook registration, saved-tensor hooks, `save_on_cpu`, mutation allowances, and `_engine_run_backward` |
| `forward_ad.py` | Provides dual-level management, `make_dual`, and `unpack_dual` for forward-mode automatic differentiation |
| `anomaly_mode.py` | Provides anomaly detection context managers that enable forward traceback recording and NaN checks in backward |

## Public Interface

The package exports `Variable`, `Function`, `backward`, `grad`, `gradcheck`, `gradgradcheck`, `detect_anomaly`, `set_detect_anomaly`, `no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode`, `set_multithreading_enabled`, and `enforce_grad_layout_policy`. Submodules expose `torch.autograd.functional.{vjp,jvp,jacobian,hessian,hvp,vhp}`, `torch.autograd.graph.{Node,GradientEdge,saved_tensors_hooks,save_on_cpu,register_multi_grad_hook}`, and `torch.autograd.forward_ad.{dual_level,make_dual,unpack_dual}`. Custom operation authors subclass `Function`, implement static `forward` and `backward` or `vjp`, optionally implement `jvp` and `vmap`, save tensors with `ctx.save_for_backward`, and invoke the operation with `Function.apply`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Calls `_engine_run_backward`, uses C++ graph nodes exposed to Python, and relies on engine execution for `backward` and `grad` |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Uses vmap and forward-AD predispatch helpers for batched gradients, `functional.jacobian(vectorize=True)`, and dual tensors |
| [torch/fx](torch/fx/ADR.md) | depends-on | Autocast/export and graph helpers interact with proxy and dispatch modes that preserve graph semantics under tracing |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | Neural-network modules use `Parameter.requires_grad`, `.backward()`, hooks, grad mode, and functional gradients during training |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimizers consume `.grad` buffers produced by `backward` and run differentiable steps under autograd control |

## Runtime Behaviour

`backward` and `grad` normalize tensor or `GradientEdge` inputs, validate gradient shapes and dtypes in `_make_grads`, honor `__torch_function__`, choose `retain_graph` from `create_graph` when unspecified, and call `_engine_run_backward` with either accumulation into leaves or returned gradients. `grad_mode.py` changes thread-local C++ grad state through `torch._C._set_grad_enabled` or inference-mode guards, so forward operations either record autograd edges or skip graph construction. `Function.apply` routes user-defined operations through the C++ `_FunctionBase`, while `FunctionCtx.save_for_backward` stores tensors for later `backward` access and participates in saved-tensor hooks. `functional.py` composes these primitives: `vjp` calls `grad`, `jvp` uses the double-backward trick unless the forward-mode path is selected, and vectorized Jacobian paths use vmap over basis vectors.

## Performance Profile

The fastest user path is thin: `backward` performs Python validation once and then transfers execution to the C++ engine, so scheduling, dependency tracking, and tensor additions occur outside Python. `no_grad` and `inference_mode` remove graph-construction work during inference; `inference_mode` also disables view tracking and version-counter bumps as documented in `grad_mode.py`. `functional.jacobian(vectorize=True)` reduces Python loop overhead by issuing a single batched `autograd.grad` through vmap, but the file explicitly marks the feature experimental and warns about performance cliffs. `gradcheck` intentionally prioritizes correctness diagnostics over speed because it perturbs inputs, computes analytical and numerical Jacobian entries, and handles complex, sparse, and nondeterministic cases.

## Design Rationale

The Python package separates user ergonomics from engine mechanics. It accepts flexible Python inputs, dictionaries, tensor subclasses, hooks, and decorators while preserving a single C++ execution path for the actual backward graph from Chapter 05. `Function` keeps custom operation state in a context object because Python users need to save tensors and non-tensor metadata without authoring C++ `Node` subclasses. The functional APIs live here because they are transformations over arbitrary Python callables, not new tensor kernels.
