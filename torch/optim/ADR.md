# Architecture Decision Record: Optimizers (torch.optim)

## Architectural Role

**torch.optim** is PyTorch's optimizer library — the module that implements gradient-based optimization algorithms for updating model parameters during training. It takes gradients computed by autograd and applies update rules to parameters. Multiple optimization algorithms are provided: SGD, Adam, AdamW, RMSprop, LBFGS, and 20+ others.

**Location**: `torch/optim/` | **Language**: Python | **Dependencies**: autograd, torch.nn, torch

## Responsibilities

**torch.optim owns**:
- Optimizer base class (`Optimizer`) defining interface
- Specific optimizer implementations (SGD, Adam, AdamW, RMSprop, LBFGS, Adadelta, Adagrad, etc.)
- Optimizer state management (momentum buffers, running averages, etc.)
- Parameter group management (different learning rates for different parts of model)
- Learning rate scheduling (`optim.lr_scheduler`)
- Checkpoint/restore interface (`state_dict()`, `load_state_dict()`)
- Gradient normalization utilities

**torch.optim does not own**:
- Gradient computation (autograd owns that)
- Parameter definitions (torch.nn owns that)
- Tensor operations (ATen owns those)
- Model training loops (user code owns that)

## Dependencies

### Inbound Dependencies
- **Training loops** call `optimizer.step()` to update parameters
- **Model code** passes `model.parameters()` to optimizer constructor
- **Checkpoint/restore** code uses `optimizer.state_dict()` and `load_state_dict()`

### Outbound Dependencies
- **torch.nn** accessed through `model.parameters()` to get parameters
- **autograd** accesses `.grad` attribute of parameters (computed by backward pass)
- **torch** for Tensor type and operations

## Trade-offs and Design Decisions

### 1. Parameter Groups for Selective Learning Rates
**Decision**: Optimizer takes list of parameter groups, each with separate learning rate and hyperparameters.

**Rationale**: 
- Transfer learning: fine-tune pretrained base at low LR, train new head at high LR
- Discriminative learning rates: different rates for different layers
- Custom learning rate schedules per layer group

**Example**: 
```python
optimizer = optim.SGD([
    {'params': model.base.parameters(), 'lr': 0.001},
    {'params': model.head.parameters(), 'lr': 0.01}
])
```

**Trade-off**: More complex API; single flat parameter list simpler but less flexible.

### 2. Optimizer State Separate from Parameters
**Decision**: Optimizer maintains separate state dict (momentum buffers, second moments) rather than storing on parameters.

**Rationale**: 
- Parameters should only contain model weights; optimizer state is temporary
- Checkpoint size: model state_dict is small and portable; optimizer state much larger
- Enables switching optimizers: can load model parameters into different optimizer

**Evidence**: optimizer.py's `self.state` dict stores per-parameter state.

**Trade-off**: State duplication; user must manage both model and optimizer checkpoints.

### 3. Base Class Interface with Subclass Implementation
**Decision**: `Optimizer` base class defines interface; each algorithm (SGD, Adam, etc.) implements `step()` method.

**Rationale**: 
- Uniform interface: all optimizers have `zero_grad()`, `step()`, `state_dict()`, `load_state_dict()`
- Easy to swap optimizers in training loop
- Centralized parameter/state management in base class

**Evidence**: optimizer.py (base) vs. sgd.py, adam.py, etc. (implementations)

### 4. zero_grad() Before Step
**Decision**: Users must call `optimizer.zero_grad()` before each training step to clear previous gradients.

**Rationale**: 
- Explicit is better than implicit (Python philosophy)
- Enables gradient accumulation: don't call zero_grad() to accumulate gradients over multiple backward passes
- Some users intentionally accumulate gradients; API should allow it

**Trade-off**: Users can forget to call zero_grad(), causing bugs. Could auto-clear, but that removes flexibility.

### 5. State Dict Serialization for Checkpointing
**Decision**: Both model and optimizer support `state_dict()` and `load_state_dict()` for checkpoint/restore.

**Rationale**: 
- Enables resuming training from checkpoint with exact same optimizer state
- Momentum buffers, running averages must be persisted to resume correctly
- Checkpoint files human-readable JSON/pickle format

**Evidence**: All optimizers implement state_dict()

### 6. Closure Parameter for Line Search
**Decision**: Some optimizers (LBFGS, some SGD configurations) accept optional `closure` parameter in `step()`.

**Rationale**: 
- Some algorithms need to re-evaluate loss/gradients at different parameter values
- Closure is callable that resets gradients, re-evaluates loss, returns loss value
- Enables optimization algorithms that require multiple function evaluations

**Trade-off**: Adds complexity; most users don't use closures.

### 7. Defaults Dict for Algorithm Hyperparameters
**Decision**: Each optimizer subclass defines `defaults` dict with algorithm-specific hyperparameters (momentum for SGD, betas for Adam, etc.).

**Rationale**: 
- Centralizes algorithm parameters in one place
- Parameter groups inherit defaults unless overridden
- Easy to see what parameters an optimizer accepts

**Evidence**: sgd.py has `defaults = dict(lr=required, momentum=0, ...)`, adam.py has different defaults.

## Extension Boundaries

**Custom optimizers**: Subclass `Optimizer`, implement `step()` method, define defaults dict.

**Learning rate scheduling**: Use `optim.lr_scheduler` classes to modify learning rates during training (e.g., exponential decay, cosine annealing).

**Distributed optimization**: Wrap optimizer with `torch.distributed` communication to aggregate gradients across processes.

**Mixed precision optimization**: Use `torch.cuda.amp.GradScaler` to scale gradients for FP16 training while optimizer updates in FP32.

## Runtime Implications

### Initialization
- Optimizer initialization: register parameters, initialize state dicts
- Cheap operation (~milliseconds for large models)
- State initialization happens lazily on first `step()`

### Training Loop Lifecycle
1. `optimizer.zero_grad()` — set parameter.grad to 0
2. Forward pass — compute outputs
3. `loss.backward()` — autograd computes gradients into parameter.grad
4. `optimizer.step()` — update parameters using gradients
5. Repeat

### State Management
- First `step()` call: allocate state buffers (momentum, running averages)
- Subsequent steps: update state buffers and parameters
- `state_dict()`: returns {param_group_idx: {state_name: tensor, ...}}

### Memory
- Optimizer state can be as large as model parameters (e.g., Adam needs 2x model size: first and second moment estimates)
- State persisted in checkpoints; can be large for distributed training

### Concurrency
- **Not thread-safe** for concurrent `step()` calls on same optimizer
- **Safe** for reading completed state (e.g., for logging)
- Distributed optimizers handle multi-process synchronization

### Lifecycle
- Optimizer lives for entire training run (or until re-created)
- State accumulated during training; should be periodically checkpointed
- On resume: load both model and optimizer state to continue from exact state

## Performance Implications

### Known Hotspots
1. **Step execution time**: Depends on algorithm (SGD is O(N), Adam is O(N), LBFGS is O(N^2) or worse)
2. **State dict serialization**: Dominated by I/O, not computation
3. **Gradient access**: Each optimizer step iterates over all parameter groups and parameters

### Optimization Opportunities
1. **Fused kernels**: Some optimizers (Adam) can be fused into single GPU kernel for better cache locality
2. **Distributed gradient averaging**: Overlap gradient communication with computation
3. **Gradient accumulation**: Update less frequently to reduce communication overhead
4. **Mixed precision**: Use lower precision (FP16) for forward/backward, FP32 for optimizer

### Synchronization Costs
- Single-GPU: negligible
- Multi-GPU (DataParallel): synchronization before step() to average gradients
- Multi-node (DistributedDataParallel): all-reduce for gradient averaging, can be expensive on slow networks

## Ownership Boundaries

**torch.optim owns**:
- Optimizer algorithms and update rules
- State management and parameter group handling
- Checkpoint interface

**torch.optim delegates to**:
- autograd (for gradient computation)
- torch.nn (for accessing parameters)

**Parent/peer systems own**:
- Training loop (calls optimizer.step, manages learning rate scheduling)
- torch.optim.lr_scheduler (learning rate scheduling)
- torch.cuda.amp (mixed precision training)
- User code (decides update frequency, gradient accumulation strategy)
