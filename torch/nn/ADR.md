# `torch/nn`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nn` is the public Python neural-network package described in book Chapter 10, "Neural Network Modules". It gathers stateful modules, stateless functional operators, learnable `Parameter` and persistent `Buffer` tensor subclasses, initialization utilities, convolution gradient helpers, C++ frontend interop, and package-level imports into the `torch.nn` namespace. The directory provides the user-facing model-building API; concrete layer implementations live primarily in `torch/nn/modules` and delegate numerical work to ATen operators.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Populates the `torch.nn` namespace with `Parameter`, module classes, functional APIs, init utilities, parallel wrappers, and `factory_kwargs` |
| `parameter.py` | Defines `Parameter`, `Buffer`, `UninitializedParameter`, `UninitializedBuffer`, lazy materialization, and special tensor-subclass registration flags |
| `functional.py` | Provides stateless layer functions such as convolution, activation, normalization, pooling, dropout, loss, attention, and embedding operations |
| `init.py` | Provides in-place parameter initialization functions such as uniform, normal, Xavier, Kaiming, orthogonal, sparse, and trunc-normal initializers |
| `grad.py` | Provides explicit convolution input/weight gradient helpers backed by `torch.ops.aten.convolution_backward` |
| `cpp.py` | Wraps C++ frontend modules with `ModuleWrapper` and dynamic ordered-dict views for parameters, buffers, and submodules |
| `_reduction.py` | Normalizes legacy `size_average`/`reduce` arguments into modern reduction strings used by loss modules and functions |

## Public Interface

`torch.nn` exports `Module` and all layer classes imported from `torch.nn.modules`, `Parameter`, `Buffer`, lazy uninitialized variants, `functional`, `init`, `grad`, `utils`, `parallel`, and `DataParallel`. `factory_kwargs` canonicalizes `device`, `dtype`, `memory_format`, and nested `factory_kwargs` before layer constructors pass them to tensor factory calls. Users either instantiate stateful modules such as `nn.Linear` and `nn.Conv2d`, call stateless functions such as `nn.functional.linear`, create parameters with `nn.Parameter`, or initialize existing tensors through `nn.init` functions.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn/modules](torch/nn/modules/ADR.md) | depends-on | Imports and re-exports `Module`, layer classes, containers, losses, activations, convolutions, recurrent modules, and transformer modules |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `Parameter` defaults to `requires_grad=True`, module training examples use `backward`, and initialization functions run under `torch.no_grad()` |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Functional operators call `torch` and `torch.ops.aten` kernels for convolution, activation, loss, normalization, pooling, attention, and gradient helpers |
| [torch/csrc/api](torch/csrc/api/ADR.md) | depends-on | `cpp.py` adapts C++ frontend modules to the Python `nn.Module` protocol |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimizers consume `model.parameters()` from `nn.Module` trees and update `Parameter` tensors |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | `torch.compile` traces module calls and functional operations exported through this package, as Chapter 10 describes |

## Runtime Behaviour

Importing `torch.nn` loads the package namespace by re-exporting module classes from `torch.nn.modules`, stateless functions from `functional.py`, and initialization utilities from `init.py`. Assigning a `Parameter` from `parameter.py` to a `Module` registers it in the module's `_parameters` dictionary because `Parameter` is a tensor subclass with parameter identity, while `Buffer` marks persistent non-parameter state. `functional.py` functions validate Python arguments and then call `torch` operators or `_VF` variable functions, so autograd records the underlying ATen operations when grad mode is enabled. `init.py` mutates tensors in place inside `torch.no_grad()` wrappers so initialization does not create backward history.

## Performance Profile

The `torch.nn` package itself adds little runtime overhead because top-level imports and wrappers delegate heavy computation to ATen kernels. The stateless functional API avoids module attribute lookups and is the form compiler passes prefer when inlining model computations, which matches Chapter 10's discussion of functional conversion. `Parameter.__new__` uses `torch.Tensor._make_subclass` for standard tensors, so parameters retain tensor storage and dispatch behavior without wrapping every operator. `grad.py` computes convolution gradients by calling `aten.convolution_backward` directly with output masks, avoiding full autograd graph traversal when callers need only one explicit convolution gradient.

## Design Rationale

`torch.nn` separates public namespace composition from concrete layer implementation. This keeps `nn.Linear`, `nn.Conv2d`, and other modules discoverable at `torch.nn` while their source stays organized by layer family under `torch/nn/modules`. `Parameter` exists because ordinary tensors assigned as Python attributes should not automatically become trainable model state; explicit parameter and buffer subclasses make registration predictable. The functional API exists beside modules so users, transforms, and compilers express the same operations with explicit state.
