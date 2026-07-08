# `torch/nn/modules`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nn/modules` implements the stateful neural-network module system and standard layer catalog described in book Chapter 10. It owns `Module`, parameter and buffer registration, hook execution, state dict serialization, containers, and concrete modules for linear layers, convolutions, recurrent networks, transformers, activations, normalization, pooling, dropout, embeddings, and losses. The modules hold state and metadata while their `forward` methods delegate mathematical work to `torch.nn.functional`, `_VF`, or ATen operators.

## Key Files

| File | Purpose |
|---|---|
| `module.py` | Defines `Module`, registration dictionaries, `__setattr__`, `__call__`, hooks, state dict save/load, traversal, device/dtype conversion, and parameter iteration |
| `__init__.py` | Imports the public layer catalog from individual module files into `torch.nn.modules` |
| `linear.py` | Defines `Identity`, `Linear`, `Bilinear`, lazy linear layers, parameter initialization, and `F.linear` delegation |
| `conv.py` | Defines N-dimensional convolution and transposed convolution modules, shape/group validation, padding handling, and `F.conv*d` delegation |
| `rnn.py` | Defines `RNNBase`, `RNN`, `LSTM`, `GRU`, cells, flat-weight management, input/hidden validation, and `_VF` recurrent calls |
| `transformer.py` | Defines reference `Transformer`, encoder/decoder stacks, encoder/decoder layers, masks, and composition from attention, linear, dropout, and layer norm modules |
| `activation.py` | Defines activation modules and `MultiheadAttention`, including optimized scaled-dot-product and nested-tensor inference fast paths |
| `loss.py` | Defines loss modules, reduction handling, weighted-loss buffers, and calls into `torch.nn.functional` loss functions |
| `container.py` | Defines `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, and `ParameterDict` while preserving registration semantics |
| `batchnorm.py` | Defines batch normalization modules, affine parameters, running-stat buffers, versioned load compatibility, and `F.batch_norm` calls |

## Public Interface

The directory exports `Module` plus layer classes such as `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose*`, `RNN`, `LSTM`, `GRU`, `Transformer`, `TransformerEncoder`, `TransformerDecoder`, `MultiheadAttention`, `ReLU`, `GELU`, `BatchNorm*`, `LayerNorm`, `Dropout`, `Embedding`, `CrossEntropyLoss`, `MSELoss`, and the container classes. `Module` exposes `forward`, `__call__`, `register_parameter`, `register_buffer`, `add_module`, hook registration, `state_dict`, `load_state_dict`, `parameters`, `named_parameters`, `buffers`, `named_modules`, `train`, `eval`, and device/dtype conversion methods. Individual modules expose constructor arguments as attributes, implement `forward`, and provide `extra_repr` for structured printing.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | Imports `Parameter`, `Buffer`, initialization utilities, functional operators, reduction helpers, and package-level typing conventions |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Parameters require gradients, module backward hooks attach to `grad_fn`, and tensor operations inside `forward` build backward graphs |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Functional calls dispatch to native kernels for linear algebra, convolution, attention, normalization, pooling, losses, and elementwise activations |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | FX traces `Module` calls, containers, and functional operations when building graph representations of models |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo inlines `Module.__call__`, `forward`, hooks, and parameter access during `torch.compile` tracing |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimizers iterate parameters from module trees and use parameter-group names derived from `named_parameters` |

## Runtime Behaviour

`Module.__init__` installs dictionaries for parameters, buffers, submodules, hooks, and load/save hooks directly with `super().__setattr__` to avoid registration overhead during construction. `Module.__setattr__` routes `Parameter` assignments to `_parameters`, `Module` assignments to `_modules`, and `Buffer` or registered buffer names to `_buffers`, removing the same name from conflicting registries. `Module.__call__` enters `_call_impl`; it calls `forward` directly when no local or global hooks exist and otherwise runs forward pre-hooks, backward-hook setup, `forward`, forward hooks, backward-output hook setup, and always-call hook cleanup. `state_dict` recursively walks `_modules`, saves parameters and persistent buffers with dotted prefixes, records version metadata, and lets registered hooks adjust the result.

## Performance Profile

The no-hook fast path in `_call_impl` returns `forward_call(*args, **kwargs)` immediately, so ordinary module invocation adds minimal Python overhead beyond the method call described in Chapter 10. Layer `forward` implementations are thin: `Linear.forward` calls `F.linear`, `Conv1d.forward` calls `_conv_forward` and then `F.conv1d`, activation modules call their matching `F.*` functions, and losses call `F.*_loss`. Recurrent modules maintain `_flat_weights` and weak references so backends use packed weight layouts and refresh them only when parameter objects change. `MultiheadAttention` documents and checks optimized inference constraints so self-attention uses scaled-dot-product attention and nested-tensor fast paths when training, autograd, masks, and autocast settings allow it.

## Design Rationale

`Module` acts as the composition primitive because models need a tree that carries trainable state, non-trainable state, hooks, serialization names, and mode flags together. The package implements many layer families as small Python state holders over functional kernels, which keeps numerical implementations centralized in ATen while preserving Python ergonomics. Containers register children rather than storing raw lists or dictionaries so transformations, `state_dict`, `to()`, and `parameters()` see the same module tree. Hook support lives in `Module.__call__` instead of individual layers so all modules share one execution protocol.
