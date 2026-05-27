# Architecture Decision Record: torch.optim

## Architectural Role

The `torch.optim` subsystem implements **parameter optimization** — the mechanisms by which neural networks learn. Given a loss value and computed gradients, optimizers update model parameters to minimize loss. This subsystem translates high-level training logic (forward pass, backward pass, step) into concrete parameter updates using various algorithms (SGD, Adam, RMSprop, etc.).

Key insight: Optimizers are **orchestrators**, not algorithm implementers. The base class manages parameter groups, state persistence, and the optimization loop; each algorithm subclass implements only the per-parameter update logic.

## Responsibilities

### What This Subsystem Owns

1. **Base Optimizer Class and Lifecycle** (`optimizer.py`)
   - Constructor: parse parameters into parameter groups, apply defaults
   - `zero_grad()`: clear accumulated gradients before each iteration
   - `step()`: invoke algorithm-specific update logic
   - State management: maintain per-parameter optimizer state (momentum buffers, Adam moments, etc.)
   - Hooks: pre-step and post-step instrumentation

2. **Optimizer Algorithms** (various files)
   - SGD (`sgd.py`): basic stochastic gradient descent with momentum
   - Adam (`adam.py`): adaptive momentum estimation
   - AdamW (`adamw.py`): Adam with decoupled weight decay
   - RMSprop (`rmsprop.py`): root-mean-square propagation
   - LBFGS (`lbfgs.py`): quasi-Newton method
   - Other variants: Adagrad, AdaDelta, ASGD, RAdam, Nadam, Adamax
   - Sparse variants: SparseAdam for embeddings

3. **Parameter Group Management**
   - Enable per-layer control: different learning rates, momentum, weight decay per group
   - Hyperparameter merging: group-specific values override defaults
   - State per group: each parameter carries its own state dict

4. **Learning Rate Scheduling** (`lr_scheduler.py`)
   - Adaptive learning rate adjustment based on epoch or metric
   - StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, etc.
   - Integration with optimizer: modifies `param_groups[i]['lr']` after each step or epoch

5. **State Persistence** (`optimizer.py`)
   - `state_dict()`: serialize optimizer state (per-parameter auxiliary state)
   - `load_state_dict()`: restore optimizer state for resuming training
   - Checkpoint format: maps parameter names to state dicts

### What This Subsystem Does NOT Own

- **Gradient Computation**: torch.autograd computes gradients
- **Loss Function Definition**: User defines the loss
- **Model Parameters**: torch.nn owns parameters; optimizer only reads and updates them
- **Parameter Grouping Strategy**: User decides which parameters go in which group
- **Distributed Gradient Synchronization**: torch.distributed handles AllReduce; optimizer assumes gradients are ready

## Dependencies

### Upstream Dependencies (What Uses This)

- **User Training Code**: Training loops call `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- **Learning Rate Schedulers**: External code uses lr_scheduler to adjust learning rate
- **Checkpointing**: User code calls `state_dict()` to save training state
- **Fine-tuning Workflows**: Different layers get different learning rates via parameter groups

### Downstream Dependencies (What This Uses)

- **torch.nn**: `optimizer` receives `model.parameters()` or parameter groups
- **torch.autograd**: Reads `.grad` attributes of parameters to compute updates
- **torch.tensor**: Modifies parameter tensors in-place during updates

### Dependency Direction

```
User Training Loop
    ↓
optimizer.zero_grad()
    ↓
model(x) → loss
    ↓
loss.backward() (populates param.grad via autograd)
    ↓
optimizer.step() (reads param.grad, updates param)
```

## Trade-offs and Design Decisions

### Parameter Groups vs. Multiple Optimizers

**Decision**: Use parameter groups (one optimizer with multiple groups) rather than multiple optimizer instances.

**Trade-off**:
- ✅ **Advantage**: Single state dict, coordinated step(), simpler checkpoint format
- ✅ **Advantage**: Enables fine-grained control without optimizer proliferation
- ❌ **Disadvantage**: Less flexible; algorithms must work within single-step framework
- ❌ **Disadvantage**: Complex constructor if many groups needed

**Evidence**: `Optimizer.__init__()` (lines 377-405 of optimizer.py) parses both flat iterables and list of groups; each group in `param_groups` can have different hyperparameters.

### State in Parameter vs. Global Registry

**Decision**: Optimizer state is keyed by parameter tensor identity and stored in `self.state[param]`.

**Trade-off**:
- ✅ **Advantage**: State follows parameter identity; handles parameter reuse naturally
- ✅ **Advantage**: Clear ownership: each parameter has associated state
- ❌ **Disadvantage**: If parameter tensor is recreated (different object), old state lost
- ❌ **Disadvantage**: Debugging requires understanding parameter identity

**Evidence**: `Optimizer.state` (line ~420 of optimizer.py) is `defaultdict(dict)`; keyed by parameter tensor.

### Zero_grad with set_to_none Option

**Decision**: Two modes for clearing gradients: set `.grad = None` (fast) or `.grad.zero_()` (preserves tensor).

**Trade-off**:
- ✅ **Advantage**: `set_to_none=True` faster and uses less memory
- ✅ **Advantage**: `set_to_none=False` preserves tensor for code expecting non-None `.grad`
- ❌ **Disadvantage**: Users must choose; unclear when each is appropriate
- ❌ **Disadvantage**: Default changed in PyTorch 2.0; potential for upgrade breakage

**Evidence**: `zero_grad()` method (lines 1025-1087 of optimizer.py) has `set_to_none` parameter; default is True in current versions.

### Algorithm Isolation via Subclassing

**Decision**: Each optimizer algorithm is a subclass of `Optimizer` implementing only the per-parameter update step.

**Trade-off**:
- ✅ **Advantage**: Clear separation of concerns; common logic in base, algorithm-specific in subclass
- ✅ **Advantage**: Easy to add new algorithms without modifying existing code
- ❌ **Disadvantage**: Subclassing required; no simple functional API for custom algorithms
- ❌ **Disadvantage**: Some algorithms (LBFGS) don't fit the mold cleanly

**Evidence**: `SGD` (sgd.py), `Adam` (adam.py) are independent subclasses; each overrides `step()` method.

## Extension Boundaries

### Implementing Custom Optimizers

**Boundary**: Subclass `Optimizer` and implement `step()` method.

```python
class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, **kwargs):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state if first step
                if len(state) == 0:
                    state['step'] = 0
                
                # Apply update
                p.data.add_(grad, alpha=-group['lr'])
                state['step'] += 1
        
        return loss
```

**Evidence**: Documented in PyTorch tutorials; `Optimizer` base class provides infrastructure.

### Using Parameter Groups

**Boundary**: Pass list of dicts to optimizer constructor, each with `params` key and hyperparameter overrides.

```python
optimizer = torch.optim.SGD([
    {'params': model.encoder.parameters(), 'lr': 0.001},
    {'params': model.decoder.parameters(), 'lr': 0.01, 'momentum': 0.95},
], momentum=0.9)
```

**Evidence**: `Optimizer.__init__()` processes both flat iterables and list of groups; hyperparameters merged via `add_param_group()` method.

### Adding Hooks

**Boundary**: Register hooks to execute before or after each step.

```python
def hook_fn():
    print("About to step")

optimizer.register_step_pre_hook(hook_fn)
```

**Evidence**: `Optimizer.register_step_pre_hook()` and `register_step_post_hook()` APIs (lines ~470).

## Runtime Implications

### Lifecycle and Initialization

1. **Construction**: Parse parameters into groups, initialize state dict (empty initially)
2. **Training Loop Iteration**:
   a. `zero_grad()`: clear previous gradients
   b. `forward()`: compute loss (populates `param.grad` via autograd)
   c. `backward()`: compute gradients via autograd
   d. `step()`: apply algorithm-specific updates
3. **Checkpoint**: Serialize `optimizer.state_dict()`
4. **Resume**: Load `state_dict()` to continue training with correct state

### Concurrency Behavior

**Thread Safety**:
- **Not Thread-Safe**: Multiple threads calling `step()` simultaneously will corrupt state
- **Evidence**: No locking in optimizer implementation; shared mutable state
- **Guidance**: Single-threaded training per optimizer or use locks

**Distributed Training**:
- **AllReduce Integration**: Gradients synchronized post-backward via `torch.distributed.all_reduce()`
- **Evidence**: `step()` assumes gradients ready; no built-in synchronization
- **Coordination**: DistributedDataParallel handles AllReduce; optimizer works with synchronized gradients

### Failure Behavior

1. **NaN Gradients**: Propagate to parameters; step produces NaN parameters
2. **Uninitialized Gradients**: If `param.grad is None`, skipped by optimizers (lines ~1000 of sgd.py, adam.py)
3. **Out-of-Memory**: Large state dicts (Adam with momentum) can exhaust memory
4. **Learning Rate Zero**: Parameters don't update (learning rate multiplied by zero)

**Evidence**: Optimizers check `if p.grad is None: continue` before applying updates.

## Performance Implications

### Known Hotspots

1. **Parameter Iteration**: Each `step()` iterates all parameter groups and parameters
2. **State Lookup**: `self.state[param]` lookup by tensor identity
3. **In-Place Operations**: Parameter updates are in-place `p.data.add_()` calls

### Allocation Patterns

- **State Per Parameter**: Adam allocates two buffers per parameter (exp_avg, exp_avg_sq); linear in parameter count
- **No Allocation During Step**: Buffers pre-allocated in first step; subsequent steps reuse

### Synchronization Costs

- **Distributed Sync**: AllReduce happens outside optimizer; optimizer assumes synchronized gradients
- **No Internal Sync**: Optimizer step is single-threaded per thread

## Ownership Boundaries

### State Owned by Optimizer

1. **Parameter State**: `self.state[param]` dict for each parameter (momentum buffers, Adam moments, etc.)
2. **Parameter Group Registry**: `self.param_groups` list with hyperparameters
3. **Defaults**: `self.defaults` dict of default hyperparameters

### State Borrowed from Model/Autograd

1. **Parameters**: References to parameter tensors; data owned by torch.nn
2. **Gradients**: `.grad` attributes of parameters; owned by autograd
3. **Loss**: Scalar loss value computed by user; optimizer only uses for line-search (LBFGS)

### State Owned by Users

1. **Parameter Groups**: User decides which parameters go in which group
2. **Hyperparameters**: User sets learning rate, momentum, weight decay, etc.
3. **Step Timing**: User calls `step()` when ready

## Key Implementation Files

| File | Purpose |
|---|---|
| `optimizer.py` | Base `Optimizer` class, parameter groups, state management |
| `sgd.py` | SGD algorithm with momentum and weight decay |
| `adam.py` | Adam algorithm with exponential moving averages |
| `adamw.py` | Adam with decoupled weight decay |
| `rmsprop.py` | RMSprop algorithm |
| `lbfgs.py` | LBFGS quasi-Newton method |
| `lr_scheduler.py` | Learning rate scheduling (100+ classes) |
| `_stateless.py` | Functional optimizer implementations |
| `swa_utils.py` | Stochastic weight averaging utilities |
| `_functional.py` | Functional (pure) versions of update steps |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/optim/optimizer.py` and algorithm implementations — checked for class hierarchy, state management, update logic
2. **Book Chapter**: Chapter 07 "Optimization: Gradient Descent and Beyond" provides detailed understanding of parameter update mechanisms
3. **Code Flow**: Traced from user `optimizer.step()` call through parameter group iteration to per-parameter updates

Last Verified: 2026-05-27
