# `torch/optim`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/optim` implements parameter update algorithms for tensors produced by `torch.nn.Module.parameters()`, as covered by book Chapter 10's training loop architecture. It owns the base `Optimizer` abstraction, algorithm classes such as `SGD`, `Adam`, and `AdamW`, functional update kernels, learning-rate schedulers, stochastic weight averaging utilities, sparse optimizers, foreach/fused dispatch choices, optimizer hooks, and state dict serialization. The package consumes gradients produced by autograd and mutates parameters in-place according to parameter-group options.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public optimizer namespace exporting algorithm classes, schedulers, SWA utilities, and stateless swapping helpers |
| `optimizer.py` | Defines `Optimizer`, parameter groups, state dictionaries, step hooks, state/load hooks, foreach/fused capability checks, differentiable-step grad-mode handling, and graph-capture checks |
| `sgd.py` | Implements `SGD`, momentum state initialization, Nesterov validation, AMP fused-step support, and single-tensor/foreach/fused SGD functional paths |
| `adam.py` | Implements `Adam`, lazy moment state, capturable step tensors, AMSGrad state, sparse-gradient rejection, foreach/fused/default dispatch, and decoupled weight decay option |
| `adamw.py` | Implements `AdamW` as Adam with `decoupled_weight_decay=True` and preserves that invariant during state loading |
| `lr_scheduler.py` | Implements `LRScheduler` and concrete learning-rate schedules such as `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, and `ReduceLROnPlateau` |
| `_functional.py` | Re-exports functional optimizer implementations and implements sparse Adam updates over sparse gradient values |
| `sparse_adam.py` | Provides the sparse Adam optimizer class for sparse gradients that dense Adam rejects |
| `swa_utils.py` | Provides stochastic weight averaging helpers and SWA learning-rate scheduling |

## Public Interface

The package exports `Optimizer`, `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `ASGD`, `LBFGS`, `NAdam`, `RAdam`, `Rprop`, `SparseAdam`, `Adafactor`, `Muon`, `lr_scheduler`, `swa_utils`, and `swap_in_optimizer_params_and_state`. Every optimizer accepts an iterable of tensors or parameter-group dictionaries, maintains `param_groups` and `state`, implements `step`, `zero_grad`, `state_dict`, `load_state_dict`, and hook registration inherited from `Optimizer`. Functional APIs such as `sgd`, `adam`, and `adamw` update lists of tensors and state tensors directly and are used by classes and distributed/stateless paths.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | Accepts `Parameter` tensors from modules and uses parameter groups built from model parameter iterators |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Reads `.grad` tensors produced by backward and uses grad-mode guards for differentiable optimizer steps |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Uses tensor in-place math, foreach kernels, sparse tensor operations, fused optimizer kernels, and AMP non-finite checks |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Inserts graph breaks around differentiable step mode and adapts capturable paths during compilation |
| [torch/amp](torch/amp/ADR.md) | depended-on-by | `GradScaler` calls optimizer `step`, injects `grad_scale` and `found_inf` for fused optimizers, and skips updates on non-finite gradients |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | Distributed optimizers and sharded training rely on functional optimizer APIs and optimizer state serialization |

## Runtime Behaviour

`Optimizer.__init__` validates that parameters are an iterable of tensors or dictionaries, creates `state` as a `defaultdict(dict)`, normalizes bare parameter lists into a single parameter group, and calls `add_param_group` for each group. `SGD.step` gathers parameters with gradients, gradient tensors, and optional momentum buffers, then calls `sgd` with group options and writes updated momentum buffers back to `state`. `Adam.step` performs an accelerator graph-capture health check, lazily initializes per-parameter `step`, `exp_avg`, `exp_avg_sq`, and optional `max_exp_avg_sq`, rejects sparse gradients, and calls `adam` with group flags for foreach, fused, capturable, differentiable, and decoupled weight decay. `LRScheduler` attaches to an `Optimizer`, records each group's `initial_lr`, wraps `optimizer.step` to track call order, and mutates group learning rates during `scheduler.step`.

## Performance Profile

Optimizers update many small tensors, so `optimizer.py` defaults eligible parameter lists to foreach kernels when scripting and differentiable mode do not block them; fused kernels remain opt-in unless explicitly requested. `adam.py` and `sgd.py` choose among single-tensor loops, `_foreach_*` multi-tensor kernels, and `_fused_*` kernels based on user flags, device support, dtype support, sparse gradients, and graph-capture constraints. Adam stores the step tensor on CPU when neither capturable nor fused mode is active to avoid expensive device scalar kernel launches, but it keeps step tensors on device when CUDA/XPU graph capture or fused kernels need device-resident state. `GradScaler` integration lets fused optimizers handle scaling and found-inf checks inside fused update paths, avoiding separate unscale and skip passes.

## Design Rationale

The base `Optimizer` separates parameter grouping and serialization from algorithm math so all optimizers share hooks, state dicts, graph-capture checks, and foreach/fused selection logic. Algorithm classes own user validation and lazy state initialization, while functional routines own tensor math; this split supports stateless use, distributed wrappers, compilation, and direct testing of update formulas. `AdamW` subclasses `Adam` instead of duplicating the implementation because decoupled weight decay is a mode of the same moment update machinery. Learning-rate schedulers mutate optimizer parameter groups rather than parameters so they compose with every optimizer class.
