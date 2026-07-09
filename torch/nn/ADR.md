# `torch/nn`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nn` defines the neural network module system: the `Module` base class, all standard layer implementations, the functional API, and weight-initialization utilities. It is the primary abstraction users build models with.

## Key Files

| File | Purpose |
|---|---|
| `modules/module.py` | `Module` — 3,053-line base class; owns `_parameters`, `_buffers`, `_modules` dicts; implements `forward`, `backward_hooks`, `forward_pre_hooks`, `forward_hooks`, `state_dict`, `load_state_dict`, `to`, `apply` |
| `modules/linear.py` | `Linear`, `LazyLinear`, `Bilinear`, `Identity` |
| `modules/activation.py` | All activation functions: `ReLU`, `GELU`, `Sigmoid`, `Softmax`, `MultiheadAttention`, etc. |
| `modules/conv.py` | Convolution modules: `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose*` |
| `modules/batchnorm.py` | `BatchNorm1d/2d/3d`, `LazyBatchNorm*` — running stats managed as non-gradient buffers |
| `modules/loss.py` | Loss functions: `CrossEntropyLoss`, `MSELoss`, `BCELoss`, `NLLLoss`, etc. |
| `modules/rnn.py` | `RNN`, `LSTM`, `GRU` — recurrent modules; use `torch._C._VariableFunctions` for CuDNN-backed kernels |
| `modules/transformer.py` | `Transformer`, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer` |
| `modules/container.py` | `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict` |
| `functional.py` | Stateless functional counterparts to all modules: `F.linear`, `F.conv2d`, `F.relu`, `F.cross_entropy`, etc. |
| `parameter.py` | `Parameter` (subclass of `Tensor` with `requires_grad=True` by default), `Buffer`, `UninitializedParameter` |
| `init.py` | Weight initializers: `xavier_uniform_`, `kaiming_normal_`, `orthogonal_`, `zeros_`, etc. |
| `parallel/data_parallel.py` | `DataParallel` — single-node multi-GPU replication |

## Public Interface

| Symbol | Description |
|---|---|
| `nn.Module` | Base class; `__call__` runs forward pre-hooks, `forward()`, and forward hooks; `register_parameter`, `register_buffer`, `register_module` maintain the state dicts |
| `nn.Parameter` | Tensor that registers itself in `Module._parameters` on assignment |
| `nn.Module.state_dict()` | Returns a flat `OrderedDict` of all parameter and buffer tensors |
| `nn.Module.load_state_dict(state_dict)` | Loads parameters/buffers; raises on missing or unexpected keys unless `strict=False` |
| `nn.Module.to(device_or_dtype)` | Moves all parameters and buffers in-place; returns `self` |
| `nn.Module.apply(fn)` | Recursively applies a callable to every submodule |
| `nn.functional` | Stateless operation namespace (`F.relu`, `F.linear`, `F.conv2d`, …) |
| `register_module_forward_hook(fn)` | Registers a global hook fired after every module's `forward`; returns `RemovableHandle` |
| `register_module_forward_pre_hook(fn)` | Registers a global hook fired before every module's `forward` |
| `nn.Module.register_full_backward_hook(fn)` | Registers a backward hook on the module's output gradient |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `Parameter` has `requires_grad=True`; backward hooks use `register_hook` on autograd tensors |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | All `functional.*` operations call ATen kernels via the dispatcher |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | RNN modules call `torch._C._VariableFunctions.lstm` for CuDNN kernels |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimizers receive `module.parameters()` iterators |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | `DistributedDataParallel` wraps `nn.Module` and hooks its backward pass |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | FX traces through `nn.Module.forward` and `call_module` graph nodes |
| [torch/jit](torch/jit/ADR.md) | depended-on-by | `torch.jit.script` compiles `nn.Module` subclasses to `ScriptModule` |

## Runtime Behaviour

`Module.__call__` is the hot path: it iterates `_forward_pre_hooks`, calls `self.forward(*args, **kwargs)`, then iterates `_forward_hooks` and `_forward_hooks_always_called`. Global hooks stored in module-level `OrderedDict` objects (`_global_forward_pre_hooks`, `_global_forward_hooks`) are checked via `_has_any_global_hook()` before iterating, so the fast path when no hooks are registered is a single `bool` check. Parameters and submodules are stored in `OrderedDict` fields (`_parameters`, `_modules`, `_buffers`) accessed by name; `__setattr__` intercepts assignments of `Parameter` and `Module` instances to route them to the correct dict. `to(device)` recursively calls `_apply(lambda t: t.to(device))` over all parameters and buffers; this moves tensors in-place and returns the module unchanged.

## Performance Profile

- **Allocation sites**: layer forward passes allocate intermediate tensors (e.g., `F.linear` allocates the output buffer); these are the primary allocation hot spots in the forward pass. `BatchNorm` additionally allocates per-batch statistics tensors.
- **Synchronization costs**: forward and backward hooks are invoked synchronously; global hooks in `_global_forward_hooks` iterate an `OrderedDict` on every `forward` call even if the hook list is empty (mitigated by the `_has_any_global_hook()` guard).
- **Data movement**: `Module.to(device)` moves all parameters and buffers; this is intentionally batch-moving. Individual `Parameter.to()` calls inside `load_state_dict` with `map_location` can trigger per-tensor host-device copies.
- **Redundant or repeated work**: `state_dict()` constructs a new `OrderedDict` on every call by recursing the module tree; for very large models this is measurable. `_named_members` uses `memo` sets to avoid yielding the same tensor twice when parameters are shared.

## Design Rationale

`Module` stores parameters, buffers, and submodules in separate `OrderedDict` fields (rather than a single dict) so that `parameters()`, `buffers()`, and `children()` iterators work in O(n) without filtering by type. `Parameter` is a `Tensor` subclass rather than a wrapper so that it participates transparently in ATen dispatch and autograd. Global forward/backward hooks use integer-keyed `OrderedDict` rather than a list so that `RemovableHandle.remove()` is O(1) — important when models register and deregister hooks frequently during training.
