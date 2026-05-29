# Architecture Decision Record: torch/nn/modules

## Architectural Role

`torch/nn/modules` defines the `nn.Module` hierarchy—the central abstraction for building neural networks in PyTorch. Every user-defined model inherits from `nn.Module`, making this module Runtime Critical for the user-facing API and the entire PyTorch modeling ecosystem.

## Responsibilities

- Defining `nn.Module` base class with parameter/buffer management, forward method interface, and state lifecycle
- Providing parameter tracking via `_parameters` dict (for `model.parameters()`)
- Providing buffer tracking via `_buffers` dict (for non-learnable tensors like running statistics)
- Implementing module hierarchies via `_modules` dict (for nested modules)
- Implementing `train()`/`eval()` mode toggling (affects Dropout, BatchNorm behavior)
- Providing state dict interface for serialization (`state_dict()`, `load_state_dict()`)
- Implementing module hooks for introspection (forward_pre_hooks, forward_hooks)

The module does **not** implement specific layer types (those are in `torch/nn/functional` and submodules); it provides the framework.

## Dependencies

**Inbound** (what depends on torch/nn/modules):
- All user models
- `torch/optim` for parameter iteration
- `torch/distributed` for distributed parameter handling

**Outbound** (what torch/nn/modules depends on):
- `torch/_C` (C++ bindings for performance-critical operations)
- Standard Python libraries

## Trade-offs

**Parameter discovery via `_parameters` dict**: Modules maintain a dict of named parameters discovered via attribute inspection. This enables automatic batching and distributed training but adds overhead in `__setattr__`. The alternative (explicit registration) would be simpler but require more user code.

**Module hooks for introspection**: Modules support registering hooks that execute before/after forward. This enables debugging and layer-wise analysis but adds call overhead and complicates forward pass semantics.

## Extension Boundaries

- **Custom module types**: Users can subclass `nn.Module` to create custom layers (most layers in PyTorch are defined this way).
- **Parameter and buffer registration**: Subclasses register parameters/buffers via `self.register_parameter()` and `self.register_buffer()`.

## Runtime Implications

**Parameter management**: `model.parameters()` traverses the module hierarchy and yields all registered parameters (used by optimizers).

**Mode switching**: `model.train()`/`model.eval()` recursively sets training mode on all submodules, affecting layer-specific behavior.

**State serialization**: `model.state_dict()` returns a dict of parameter/buffer names and values; `load_state_dict()` restores them. This enables checkpoint save/load.

**Hook execution**: Forward hooks are called during forward pass, adding overhead but enabling introspection and gradient manipulation.

## Performance Implications

**`__setattr__` overhead**: Every attribute assignment checks if it's a parameter/buffer/module, adding small overhead but enabling automatic discovery.

**Hierarchy traversal**: Operations like `model.parameters()` traverse the module tree, which is O(n) in the number of modules. This is acceptable for typical models (hundreds of modules) but expensive for very deep networks.

**Hook overhead**: Each forward hook adds a function call, typically 1-2 microseconds per hook. Multiple hooks on the same operation add linearly.

## Ownership Boundaries

- **nn.Module owns**: module hierarchy, parameter/buffer registry, training state
- **Submodules own**: their own parameters and implementation
- **Optimizers own**: learnable parameter state (after passing `model.parameters()`)

## Verification Points

- `torch/nn/modules/module.py` — Main Module class definition
- `torch/nn/modules/` — Directory of standard layer implementations
- `torch/optim/optimizer.py` — How optimizers iterate over parameters
