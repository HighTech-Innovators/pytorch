# `torch/nn`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nn` is PyTorch's neural-network module system. It defines the `Module` base class that manages parameters, buffers, submodules, and hooks; the standard layer library (linear, conv, normalization, attention, RNN, transformer); the functional API; and parameter/initialization utilities. It is the layer where user model structure becomes concrete runtime graphs and checkpoint state.

## Key Files

| File | Purpose |
|---|---|
| `torch/nn/modules/module.py` | `Module` base class (3053 lines): `__call__`, hooks, `to()`, `state_dict()` |
| `torch/nn/parameter.py` | `Parameter`, `Buffer` classes |
| `torch/nn/functional.py` | Stateless functional ops (`F.linear`, `F.relu`, `F.softmax`) |
| `torch/nn/modules/linear.py` | `Linear` layer |
| `torch/nn/modules/conv.py` | `Conv1d/2d/3d` and transposed convolutions |
| `torch/nn/modules/batchnorm.py` | `BatchNorm*` with running-statistics buffers |
| `torch/nn/modules/normalization.py` | `LayerNorm`, `GroupNorm` |
| `torch/nn/modules/container.py` | `Sequential`, `ModuleList`, `ModuleDict` |
| `torch/nn/modules/transformer.py` | `Transformer`, `TransformerEncoderLayer` |
| `torch/nn/init.py` | Weight-initialization schemes |

## Public Interface

`nn.Module`, `Module.__call__`, `Module.forward`, `register_parameter()`, `register_buffer()`, `register_module()`, `named_parameters()`, `state_dict()`, `load_state_dict()`, `to()`/`cpu()`/`float()`, `register_forward_hook()`, `register_forward_pre_hook()`, `register_full_backward_hook()`, `nn.Parameter`, and the layer classes (`nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Sequential`, …).

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Forward calls build the autograd graph; parameters require grad |
| `torch._C` / ATen | depends-on | Layer forward bodies call ATen operators via `torch.*`/`F.*` |
| [torch/nn/parallel](torch/nn/parallel/ADR.md) | depended-on-by | DDP/DataParallel wrap `Module` instances |
| User models | depended-on-by | Every model subclasses `nn.Module` |

## Runtime Behaviour

`Module.__init__` sets the internal dicts (`_parameters`, `_buffers`, `_modules`, and hook dicts) via `object.__setattr__` to bypass registration logic. Assigning a `Parameter`/`Module` attribute is intercepted by `__setattr__`, which routes it into the correct dict, preserving insertion order for stable state-dict keys. `model(input)` invokes `__call__` (aliased to `_wrapped_call_impl`); if `_compiled_call_impl` is set it delegates to the compiled callable, otherwise `_call_impl` takes a true fast path returning `forward(*args)` directly when no hooks are registered, or the slow path running forward pre-hooks → `forward` → forward hooks (plus backward-hook plumbing) in strict order. `_apply()` recursively walks the dicts for `to()`/device transfer, mutating tensors in place.

## Performance Profile

Every layer call enters `_call_impl` before any operator dispatch, so module-call overhead is a per-layer Python cost; registering even one hook forces the call off the fast path into the larger hook-iterating control-flow block, measurably increasing latency for hook-free CPU inference. `_parameters`/`_buffers` traversal in `named_parameters()`/`state_dict()` is proportional to model size and runs on checkpoint save/load. In-place `_apply` device transfer avoids reallocating parameter objects. The real compute cost lives below this layer in the ATen kernels the `forward` bodies call.

## Design Rationale

Intercepting `__setattr__` lets users write `self.weight = nn.Parameter(...)` naturally while auto-registering into the traversable dicts that power optimizers, serialization, and device movement. Separating `_parameters` from `_buffers` makes the "trained vs. persistent-but-not-trained" distinction explicit so `model.parameters()` feeds optimizers directly. Routing execution through `__call__` rather than `forward` provides a stable hook-injection point that user code cannot accidentally bypass. `_version` in state dict enables backward-compatible checkpoint migration. The fast/slow path split in `_call_impl` keeps hook-free inference cheap.
