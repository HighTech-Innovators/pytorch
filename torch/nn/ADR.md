# Architecture Decision Record: torch.nn

## Architectural Role

The `torch.nn` subsystem is PyTorch's **user-facing abstraction for composing neural networks**. It provides:

1. **Modular building blocks** — layer types (Linear, Conv2d, BatchNorm, ReLU, etc.)
2. **Parameter tracking and optimization** — automatic registration of learnable weights and buffers
3. **Execution lifecycle management** — forward pass orchestration, hooks, training/eval modes
4. **State persistence** — serialization and deserialization of model weights

This subsystem bridges the autograd system (which computes gradients) with domain-specific neural network semantics (layers, modules, hierarchical composition). Users never directly manipulate tensors in low-level ATen operations; instead, they compose high-level modules that internally orchestrate tensor operations.

## Responsibilities

### What This Subsystem Owns

1. **Module Base Class and Registry** (`modules/module.py`)
   - `Module.__init__()`: registry initialization for parameters, buffers, submodules, hooks
   - `Module.__setattr__()`: automatic parameter/buffer/module registration on attribute assignment
   - `Module.__call__()`: forward orchestration with hooks
   - Module traversal APIs: `parameters()`, `buffers()`, `named_modules()`, `children()`

2. **Layer Implementations** (`modules/*.py`)
   - Linear/fully connected layers (`linear.py`)
   - Convolutional layers (`conv.py`)
   - Normalization layers (`batchnorm.py`, `instancenorm.py`, `layernorm.py`)
   - Activation functions (`activation.py`)
   - Pooling layers (`pooling.py`)
   - Recurrent layers (`rnn.py`)
   - Embedding layers (`sparse.py`)
   - Container modules (`container.py` — Sequential, ModuleList, ModuleDict)

3. **Functional Operations** (`functional.py`)
   - Stateless counterparts to layer modules
   - Dense operations (200+ functions) without learnable parameters
   - Used within layer forward methods or directly in custom code

4. **Parameter and Buffer Registration** (`parameter.py`, `modules/module.py`)
   - `Parameter` class: wraps tensors that require gradients and are tracked by optimizers
   - Buffer registration protocol: non-learnable state that moves between devices
   - Life cycle: registration on assignment, removal when set to None

5. **Mode Management** (`modules/module.py`)
   - `train()` / `eval()` modes: propagated recursively through module hierarchy
   - Affects layer behavior (Dropout, BatchNorm) based on `module.training` flag

6. **Forward Hooks** (`modules/module.py`)
   - Pre-forward hooks: intercept and transform inputs before forward
   - Post-forward hooks: intercept and transform outputs after forward
   - Backward hooks: execute during backward pass
   - Global and instance-specific hook registries

7. **State Serialization** (`modules/module.py`)
   - `state_dict()`: export parameters and buffers as nested OrderedDict
   - `load_state_dict()`: restore weights from state dict
   - Checkpoint format agnostic to optimizer state

### What This Subsystem Does NOT Own

- **Gradient Computation**: Autograd system owns automatic differentiation
- **Tensor Storage**: ATen owns tensor allocation and memory layout
- **Operator Kernels**: ATen owns CPU/GPU implementations
- **Distributed Training**: torch.distributed owns synchronization; nn only cooperates
- **Optimization**: torch.optim owns parameter update logic; nn only provides `.parameters()`

## Dependencies

### Upstream Dependencies (What Uses This)

- **User Code**: Training scripts, model definitions, fine-tuning workflows
- **torch.optim**: Optimizers iterate over `model.parameters()` to update weights
- **torch.distributed**: DDP wraps modules to synchronize gradients
- **Deployment/TorchScript**: Module's forward is compiled to JIT/TorchScript
- **Checkpointing**: External code calls `state_dict()` and `load_state_dict()`

### Downstream Dependencies (What This Uses)

- **torch.autograd**: Imported by `Parameter` class; forward methods produce autograd graphs
- **torch.tensor**: All layers create and manipulate tensors
- **aten/**: Operations within layer forward methods call ATen kernels
- **torch._C**: C++ bindings for performance-critical paths (particularly in functional.py)

### Dependency Direction

```
User Code
    ↓
torch.nn.Module subclasses (Linear, Conv2d, etc.)
    ↓
torch.autograd (gradients flow through)
    ↓
ATen operators (forward/backward kernels)
```

## Trade-offs and Design Decisions

### Automatic vs. Manual Parameter Registration

**Decision**: Parameters are automatically registered when assigned as module attributes; buffers also register automatically.

**Trade-off**:
- ✅ **Advantage**: Minimal boilerplate; intuitive syntax (just assign `self.weight = Parameter(...)`)
- ✅ **Advantage**: Hard to forget to register
- ❌ **Disadvantage**: Magic behavior via `__setattr__` override; harder to debug
- ❌ **Disadvantage**: All module attributes are candidates for registration; must understand the protocol

**Evidence**: Chapter 06 shows that `Module.__setattr__` (lines 600-700) intercepts assignment and routes to `_parameters`, `_buffers`, or `_modules` depending on type. `Linear.__init__` (lines 92-98 of linear.py) shows plain attribute assignment `self.weight = Parameter(...)` without explicit registration calls.

### Training Mode Propagation

**Decision**: `train()` and `eval()` modes are recursively propagated to all submodules.

**Trade-off**:
- ✅ **Advantage**: Single call `model.train()` switches entire architecture uniformly
- ✅ **Advantage**: Clear semantics: no module can be left in inconsistent state
- ❌ **Disadvantage**: Cannot mix training and eval modes within a model easily
- ❌ **Disadvantage**: Overhead to traverse entire module tree on mode change

**Evidence**: `Module.train()` method (lines 2885-2905) shows recursive traversal via `self.children()`.

### Hook-Based Instrumentation

**Decision**: Hooks allow users to intercept forward and backward execution without modifying layer code.

**Trade-off**:
- ✅ **Advantage**: Flexible instrumentation (profiling, debugging, gradient clipping)
- ✅ **Advantage**: Separates instrumentation logic from forward logic
- ❌ **Disadvantage**: Hooks have overhead; `_call_impl()` checks for hook presence on every forward
- ❌ **Disadvantage**: Hook execution order and side effects can be subtle

**Evidence**: Chapter 06 describes hook infrastructure (lines 1805-1848 of module.py). `_call_impl()` (line 1782) checks `self._forward_hooks` before executing forward.

### Functional vs. Module Interfaces

**Decision**: Both stateful (module) and stateless (functional) versions exist for most operations.

**Trade-off**:
- ✅ **Advantage**: Functional versions useful for simple compositions and testing
- ✅ **Advantage**: Modules useful for parameter management and architectural reuse
- ❌ **Disadvantage**: Maintenance burden of dual APIs; easy to diverge
- ❌ **Disadvantage**: Users must choose between approaches

**Evidence**: `torch.nn.functional` (220+ functions) vs. `torch.nn.Linear`, `torch.nn.Conv2d`, etc.

### Module as Dictionary vs. Positional

**Decision**: Submodules accessed as attributes (e.g., `model.layer1`) or via `named_modules()`.

**Trade-off**:
- ✅ **Advantage**: Intuitive, aligns with Python object model
- ✅ **Advantage**: IDE autocompletion works
- ❌ **Disadvantage**: Hard to dynamically create/remove modules
- ❌ **Disadvantage**: Container types (ModuleList, ModuleDict) needed for dynamic cases

**Evidence**: `Sequential` preserves order via `ModuleList` internally; `ModuleDict` provides string key access.

## Extension Boundaries

### Creating New Layer Types

**Boundary**: Subclass `Module`, override `__init__()` and `forward()`.

```python
class MyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

Assignment of `Parameter` or nested `Module` instances automatically registers them.

**Evidence**: Chapter 06 shows `torch.nn.Linear` following this pattern.

### Registering Custom State

**Boundary**: Use `register_parameter()`, `register_buffer()` for non-standard state.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('running_sum', torch.zeros(10))
        self.register_parameter('custom_weight', nn.Parameter(...))
```

**Evidence**: `Module.register_buffer()` (line ~1090) and `Module.register_parameter()` (line ~1060) are public APIs.

### Adding Forward Hooks

**Boundary**: Register hooks to intercept forward/backward.

```python
def hook_fn(module, input, output):
    print(f"Forward output: {output}")
    return output

module.register_forward_hook(hook_fn)
```

**Evidence**: `Module.register_forward_hook()` (line ~1932), `register_forward_pre_hook()` (line ~1916), `register_full_backward_hook()` (line ~2000).

## Runtime Implications

### Lifecycle and Initialization

1. **Instantiation**: `Module.__init__()` creates registries (parameters, buffers, modules, hooks)
2. **Attribute Assignment**: User assignments trigger `__setattr__()` which populates registries
3. **Forward Call**: `__call__()` delegates to `_call_impl()`, which orchestrates hooks and forward
4. **Mode Switch**: `train()` / `eval()` propagates mode recursively
5. **Cleanup**: No explicit cleanup; garbage collection when module no longer referenced

### Concurrency Behavior

**Thread Safety**:
- **Unsafe**: Module state (parameters, buffers, `training` flag) not thread-safe
- **Evidence**: No locking in `Module` class; shared mutable state
- **Guidance**: Each thread should have its own module instance or use locks

**Distributed Training**:
- **DDP Integration**: `torch.nn.parallel.DistributedDataParallel` wraps modules to synchronize gradients
- **Evidence**: Module is used as-is; synchronization happens after backward

### Failure Behavior

1. **Missing Parameters**: Assignment of None removes from registry; forward should handle None buffers gracefully
2. **Shape Mismatches**: Forward calls will raise tensor operation errors (ATen level)
3. **Device Mismatch**: Operations on mismatched devices raise errors
4. **Hook Exceptions**: Exception in hook terminates forward; not caught by module layer

**Evidence**: `Linear` checks `self.bias is None` before adding (line ~120 of linear.py).

## Performance Implications

### Known Hotspots

1. **`__setattr__()` Overhead**: Every attribute assignment in module construction checks type and routes to registry
2. **Hook Iteration**: `_call_impl()` iterates all hooks on every forward; hooks not optimized if unused
3. **`state_dict()` Traversal**: Recursively walks module tree to collect parameters/buffers

### Allocation Patterns

- **No Allocation in Forward Loop**: Module setup allocates registries once; forward loop does not allocate
- **Parameter Tensors**: Allocated once during `__init__`, reused every forward

### Synchronization Costs

- **Distributed Sync**: DDP synchronizes gradients post-backward; happens outside module layer
- **No Internal Sync**: Module forward/backward not synchronized

## Ownership Boundaries

### State Owned by torch.nn

1. **Module Registries**: `_parameters`, `_buffers`, `_modules`, hook dictionaries
2. **Training Flag**: `training` attribute on each module
3. **Parameter/Buffer References**: Pointers to tensors, not the tensors themselves

### State Borrowed from Autograd

1. **Computation Graph**: Built by autograd as forward methods execute
2. **Gradients**: Stored in `.grad` attributes of leaf tensors (parameters)

### State Owned by Users/ATen

1. **Tensor Data**: Parameter tensors owned by ATen's allocator
2. **Forward Logic**: Custom forward methods defined by users
3. **Hooks**: Functions registered by users

## Key Implementation Files

| File | Purpose |
|---|---|
| `modules/module.py` | Base `Module` class, registries, hooks, lifecycle |
| `modules/linear.py` | Fully connected layers |
| `modules/conv.py` | Convolutional layers (Conv1d, Conv2d, Conv3d) |
| `modules/batchnorm.py` | Batch normalization layers |
| `modules/activation.py` | Activation functions as modules (ReLU, Sigmoid, etc.) |
| `modules/container.py` | Sequential, ModuleList, ModuleDict |
| `modules/rnn.py` | LSTM, GRU, RNN layers |
| `modules/sparse.py` | Embedding layers |
| `functional.py` | 200+ stateless functions (conv, linear, activation, etc.) |
| `parameter.py` | Parameter wrapper class |
| `init.py` | Weight initialization functions (Kaiming, Xavier, etc.) |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/nn/modules/module.py` and layer implementations — checked for class hierarchy, registration protocol, lifecycle
2. **Book Chapter**: Chapter 06 "Neural Network Modules: Composability and State" provides detailed architectural understanding
3. **Code Flow**: Traced from user module instantiation through `__init__`, attribute assignment, forward execution, and hook orchestration

Last Verified: 2026-05-27
