# Architecture Decision Record: torch/optim

## Architectural Role

`torch/optim` provides optimizer implementations (SGD, Adam, AdamW, etc.) that update model parameters based on gradients. It is Runtime Critical for training; nearly all training scripts use optimizers from this module to perform parameter updates.

## Responsibilities

- Implementing optimizer base class (`Optimizer`) with common infrastructure (parameter groups, state management, step counting)
- Implementing specific optimizers (SGD, Adam, AdamW, LAMB, RMSprop, Adagrad)
- Providing learning rate scheduling interface (`LRScheduler`)
- Supporting parameter grouping (different learning rates for different layers)
- Managing optimizer state (momentum buffers, second moments, etc.)

## Dependencies

**Inbound** (what depends on torch/optim):
- Training loops
- PyTorch Lightning, Hugging Face Transformers, and other frameworks

**Outbound** (what torch/optim depends on):
- `torch/nn/modules` for parameter iteration
- `torch/_C` for C++ bindings (in the base Optimizer)

## Trade-offs

**Parameter groups for flexibility**: Optimizers support parameter groups with different learning rates and hyperparameters, enabling fine-grained control but adding complexity.

**State dict for checkpoint save/load**: Optimizer state is exposed as a state dict (like modules), enabling easy serialization but requiring careful management during distributed training.

## Extension Boundaries

- **Custom optimizers**: Users can subclass `Optimizer` and implement custom parameter update logic.
- **Learning rate schedules**: New `LRScheduler` subclasses enable custom learning rate annealing strategies.

## Runtime Implications

**State management**: Each optimizer maintains state (momentum buffers, etc.) that grows with the number of parameters. This state is persistent across training steps.

**Parameter grouping**: Optimizers support different learning rates per parameter group, enabling strategies like layer-wise learning rate scaling.

**Gradient consumption**: Optimizers consume gradients (via `param.grad`) and clear them (via `zero_grad()`).

## Performance Implications

**Update overhead**: Per-parameter update (e.g., `param -= lr * grad`) is typically 1-2 microseconds. For models with millions of parameters, this becomes significant (tens of milliseconds per step).

**State allocation**: Optimizer state (momentum buffers, second moments) roughly doubles memory usage during training.

## Ownership Boundaries

- **Optimizer owns**: state (momentum, second moments), learning rate schedule
- **Model owns**: parameters (optimizer only updates them)

## Verification Points

- `torch/optim/optimizer.py` — Base Optimizer class
- `torch/optim/adam.py`, `torch/optim/sgd.py` — Specific optimizer implementations
- `torch/optim/lr_scheduler.py` — Learning rate scheduling
