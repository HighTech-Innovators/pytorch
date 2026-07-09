# `torch/optim`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/optim` implements gradient-based parameter update algorithms — SGD, Adam, AdamW, Adagrad, RMSprop, and others — together with the `Optimizer` base class that owns parameter groups and per-parameter state, and learning-rate schedulers in `lr_scheduler.py`.

## Key Files

| File | Purpose |
|---|---|
| `optimizer.py` | `Optimizer` base class — 3,000+ line file; owns `param_groups` list and `state` defaultdict; `step()` contract; global pre/post step hooks |
| `adam.py` | `Adam` — adaptive moment estimation; delegates to `_functional.adam` or fused/foreach kernel |
| `adamw.py` | `AdamW` — Adam with decoupled weight decay |
| `sgd.py` | `SGD` — stochastic gradient descent with optional momentum and Nesterov correction |
| `adagrad.py` | `Adagrad` — per-parameter adaptive learning rate |
| `rmsprop.py` | `RMSprop` — root-mean-square gradient normalisation |
| `lbfgs.py` | `LBFGS` — quasi-Newton optimizer; requires a `closure` to re-evaluate the loss |
| `lr_scheduler.py` | `LRScheduler` base plus all schedules: `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `CyclicLR`, etc. |
| `_functional.py` | Functional (state-free) implementations of each algorithm; called by the stateful `Optimizer` classes |
| `_multi_tensor/` | `foreach`-based multi-tensor kernel dispatch (batch-updates multiple parameters in one kernel call) |
| `_adafactor.py` | `Adafactor` — memory-efficient adaptive optimizer using factorised second-moment estimates |
| `_muon.py` | `Muon` — momentum-orthogonalised gradient update (newer algorithm) |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.optim.Optimizer` | Base class; `zero_grad()`, `step()`, `state_dict()`, `load_state_dict()`, `add_param_group()` |
| `torch.optim.Adam` | Adam optimizer; `lr`, `betas`, `eps`, `weight_decay`, `foreach`, `fused` kwargs |
| `torch.optim.SGD` | SGD; `lr`, `momentum`, `dampening`, `nesterov`, `weight_decay`, `foreach` kwargs |
| `register_optimizer_step_pre_hook(fn)` | Global hook fired before every `optimizer.step()` call |
| `register_optimizer_step_post_hook(fn)` | Global hook fired after every `optimizer.step()` call |
| `torch.optim.lr_scheduler.LRScheduler` | Base scheduler; `step()` updates `optimizer.param_groups[*]['lr']` |
| `_use_grad_for_differentiable(func)` | Decorator that enables/disables grad based on `optimizer.defaults['differentiable']` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Reads `.grad` tensors populated by `backward()`; uses `no_grad` context manager during parameter updates |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Parameter update arithmetic (`add_`, `mul_`, `addcmul_`) dispatches through ATen |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | `nn.Module.parameters()` iterators are passed to optimizer constructors |
| [torch/profiler](torch/profiler/ADR.md) | depended-on-by | `KinetoStepTracker.increment_step("Optimizer")` is called via a post-step hook registered in `torch/profiler/__init__.py` |

## Runtime Behaviour

`Optimizer.step()` iterates `param_groups`, reads `.grad` from each parameter, and calls the algorithm-specific update logic (either a functional implementation in `_functional.py` or a fused/foreach kernel). `zero_grad(set_to_none=True)` (the default since PyTorch 1.7) sets `.grad = None` rather than zeroing, which avoids an allocation-zeroing pass. The `foreach` path groups parameters by device and dtype and calls multi-tensor ATen kernels (`_foreach_add_`, `_foreach_mul_`, etc.) that operate on all matching tensors in a single kernel launch. The `fused` path (CUDA only) fuses the entire Adam update into a single CUDA kernel per dtype group. Global step hooks in `_global_optimizer_pre_hooks` and `_global_optimizer_post_hooks` are `OrderedDict`s iterated on every `step()` call.

## Performance Profile

- **Allocation sites**: the default `step()` creates no new tensors — updates are in-place via `_` variants (`add_`, `mul_`). First-step initialisation allocates state tensors (momentum buffers, second-moment estimates) per parameter.
- **Synchronization costs**: gradient tensor access after `backward()` implies a CUDA stream synchronisation if gradients were computed on GPU; this is implicit in reading `.grad.data`. The `foreach` path reduces kernel-launch overhead (one launch per dtype group vs one per parameter).
- **Data movement**: no explicit data movement in the optimizer itself; gradient tensors are already on the correct device from `backward()`. `load_state_dict` may move state tensors to match parameter devices via `_patch_func` in certain configurations.
- **Redundant or repeated work**: `zero_grad(set_to_none=False)` fills gradient tensors with zeros, which is wasted work when the gradient will be immediately overwritten by the next backward pass. `set_to_none=True` eliminates this cost.

## Design Rationale

`param_groups` is a list of dicts rather than a flat list of parameters so that different subsets of parameters can have different hyperparameters (e.g., no weight decay on biases). `state` is a separate `defaultdict` keyed by parameter tensor identity rather than being stored inside `Parameter` so that the optimizer state can be saved and loaded independently of the model, and so that non-`Parameter` tensors (e.g., optimized embeddings) can also be tracked. The `foreach` and `fused` kwargs are opt-in rather than the default to preserve backward-compatible semantics for custom parameter types.
