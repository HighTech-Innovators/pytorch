# Architecture Decision Record: Neural Network Module System (torch.nn)

## Architectural Role

**torch.nn** is PyTorch's neural network abstraction layer — the Python API for defining models, layers, loss functions, and parameter management. While autograd provides automatic differentiation and ATen provides tensor operations, torch.nn provides the **composable building blocks** for constructing neural networks. It abstracts away the details of tensor operations and gradient computation, allowing users to think at the level of layers and models.

**Location**: `torch/nn/` | **Language**: Python | **Dependencies**: autograd, ATen, torch

## Responsibilities

**torch.nn owns**:
- `nn.Module` base class and module system (module.py)
- `nn.Parameter` wrapper for trainable tensors
- Pre-built layers: `Linear`, `Conv2d`, `Conv3d`, `LSTM`, `GRU`, `Embedding`, etc.
- Loss functions: `CrossEntropyLoss`, `MSELoss`, `BCELoss`, etc.
- Container modules: `Sequential`, `ModuleList`, `ModuleDict`
- Activation functions: `ReLU`, `Tanh`, `Sigmoid`, `Softmax`, etc.
- Parameter initialization utilities
- Model serialization/deserialization
- Hooks mechanism for introspection and modification

**torch.nn does not own**:
- Autograd graph construction (autograd owns that)
- Actual tensor operations (ATen owns that)
- Gradient computation (autograd owns that)
- Optimization algorithms (torch.optim owns that)

## Dependencies

### Inbound Dependencies
- **User model code** subclasses `nn.Module` to define models
- **torch.optim** accesses parameters via `model.parameters()` for optimization
- **Training loops** call `model.train()`, `model.eval()`, forward pass
- **Model serialization** saves/loads model state

### Outbound Dependencies
- **autograd** tracks gradients through layer operations
- **ATen** provides tensor operations that layers compose
- **torch** provides Tensor type and utilities

## Trade-offs and Design Decisions

### 1. OrderedDict for Parameter Storage
**Decision**: Parameters stored by name in `OrderedDict` rather than indexed list.

**Rationale**: 
- Serialization with human-readable names (`model.weight`, not `model._params[0]`)
- Named parameter access for initialization, freezing, selective training
- Stable insertion order for reproducibility across Python versions
- Enables `.named_parameters()` API

**Alternative**: Store parameters in unnamed list (simpler) but lose naming information.

**Trade-off**: Slightly more overhead than list, but essential for usability.

### 2. Recursive Module Composition
**Decision**: Modules can contain other modules via `_modules` dict; `__setattr__` auto-registers module/parameter assignments.

**Rationale**: 
- Enables natural Python syntax: `self.layer = nn.Linear(10, 20)` automatically registers
- Recursive structure matches model architecture
- `.parameters()`, `.named_modules()`, etc. traverse hierarchy automatically
- `.train()` and `.eval()` propagate recursively to all sub-modules

**Evidence**: module.py's `__setattr__` intercepts and registers modules.

**Trade-off**: Magic in `__setattr__` can be confusing; requires user to understand module registration.

### 3. Stateful Training Mode
**Decision**: `self.training` boolean flag controls layer behavior (e.g., Dropout, BatchNorm); `.train()` and `.eval()` set recursively.

**Rationale**: 
- Differentiates training vs inference behavior
- Some layers (Dropout, BatchNorm) have different computation graphs in each mode
- Single global state simpler than per-layer flags

**Trade-off**: Mutable state; user responsible for switching modes correctly.

### 4. Parameter: Thin Wrapper Around Tensor
**Decision**: `nn.Parameter` is a subclass of `Tensor` that marks tensors as trainable parameters.

**Rationale**: 
- Parameters are tensors but with special semantics (trainable, registered in module)
- `requires_grad=True` by default for parameters
- Enables distinction between `.parameters()` (trainable) and `.buffers()` (non-trainable, but persistent state)

**Evidence**: parameter.py subclasses Tensor.

**Trade-off**: Minimal; Parameter is nearly transparent beyond marking.

### 5. Buffers: Non-Trainable Persistent State
**Decision**: Support `.register_buffer()` to mark tensors as non-trainable persistent state (e.g., running means in BatchNorm).

**Rationale**: 
- BatchNorm, similar layers need to accumulate statistics during training
- Statistics are not parameters (not updated by optimizer) but are persistent state (need serialization)
- Distinction allows clear semantic separation

**Evidence**: module.py has `_buffers` dict and `.register_buffer()` method.

### 6. Hooks Mechanism for Introspection
**Decision**: Support `register_forward_pre_hook()`, `register_forward_hook()`, `register_backward_hook()` for intercepting forward/backward passes.

**Rationale**: 
- Enables introspection: inspect activations, gradients without modifying model code
- Enables modification: intercept and modify forward/backward computation
- Enables debugging: trace which layers are called, what values flow through them

**Evidence**: module.py has `_forward_pre_hooks`, `_forward_hooks`, `_backward_hooks` dicts.

### 7. Pre-Built Layers as Module Subclasses
**Decision**: Each layer type (Linear, Conv2d, LSTM) is a `Module` subclass with its own `forward()` method.

**Rationale**: 
- Each layer has parameters (weights, biases), sub-module structure, initialization logic
- Subclassing Module ensures parameters are registered and optimizable
- Composition enables model building: `Sequential([Linear, ReLU, Linear])`

**Evidence**: modules/linear.py (Linear), modules/conv.py (Conv2d), etc.

## Extension Boundaries

**Custom layers**: Subclass `nn.Module`, define `__init__()` (register parameters), implement `forward()`.

**Custom initialization**: Register hooks or override `_init_weights()` to customize parameter initialization.

**Model export**: Implement `__repr__()` or custom serialization for checkpoint compatibility.

**Distributed models**: Use `nn.DataParallel` or `nn.parallel.DistributedDataParallel` to wrap model for multi-GPU/multi-node training.

## Runtime Implications

### Initialization
- Module creation is fast: just allocates dicts and registers parameters
- Parameter initialization (zeros, Xavier, etc.) happens in layer `__init__`
- First forward pass: no compilation or optimization; purely tensor operations

### Forward Pass
1. User calls `model(x)` or `model.forward(x)`
2. `Module.__call__()` runs forward pre-hooks, then calls `forward()`, then forward hooks
3. Each layer computes tensor operations using ATen operations
4. Autograd traces operations if `requires_grad=True`
5. Return output tensor(s)

**Overhead**: Hook execution (~microseconds) if no hooks present, negligible.

### Training Loop
1. `model.train()` sets `training=True` recursively
2. Forward pass (computation graph recorded by autograd)
3. Loss computation
4. `.backward()` triggers autograd backward pass
5. Optimizer step: `optimizer.step()` updates parameters using gradients

### State Management
- `.state_dict()` returns dict of all parameters and buffers (for checkpointing)
- `.load_state_dict()` restores parameters and buffers from checkpoint
- `.parameters()`, `.named_parameters()`, `.buffers()` expose model state

### Memory
- Parameters: ~100KB-GB depending on model size
- Buffers: ~1KB-MB (batch norm stats, etc.)
- Computation graph: recorded during forward pass, deallocated after backward

### Concurrency
- **Forward pass**: Not thread-safe if model is mutable (parameters being modified)
- **Read-only inference**: Safe for concurrent reads from multiple threads
- **Training**: Typically single-threaded per model; distributed training uses multiple processes

### Lifecycle
- Module created once per training run
- Parameters accessed/modified by optimizer during training
- Model persisted via `state_dict()` for later loading

## Performance Implications

### Known Hotspots
1. **Forward pass computation**: Dominated by ATen kernel execution, not module overhead
2. **Parameter access**: `.parameters()` traverses module hierarchy; negligible if cached
3. **Gradient computation**: Autograd traces forward pass; overhead ~10-20%

### Optimization Opportunities
1. **In-place operations**: Some layers use in-place operations to save memory
2. **Layer fusion**: TorchScript can fuse layers for deployment
3. **Quantization**: `torch.quantization` can reduce precision for inference
4. **Model pruning**: Remove unused parameters to reduce model size

### Synchronization Costs
- Module itself is synchronous
- CUDA operations asynchronous; synchronization at optimizer step

## Ownership Boundaries

**torch.nn owns**:
- Module class hierarchy and system
- Parameter and buffer registration
- Hooks mechanism
- Pre-built layer implementations
- Loss function implementations
- Initialization utilities

**torch.nn uses**:
- autograd for gradient tracking
- ATen for tensor operations
- torch for type system

**Parent/peer systems own**:
- User model code (subclasses nn.Module)
- torch.optim (optimizes parameters using gradients)
- Training loop (calls module.forward, .backward, optimizer.step)
- torch.utils.data (provides data loading)
