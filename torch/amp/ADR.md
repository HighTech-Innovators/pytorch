# Architecture Decision Record: torch/amp

## Architectural Role

`torch/amp` provides automatic mixed precision (AMP) training infrastructure, enabling models to use float16 or bfloat16 for computation while maintaining float32 for gradient accumulation. AMP reduces memory usage and improves speed on supporting hardware, making it important for efficient training.

## Responsibilities

- Implementing autocast context manager for automatic dtype selection
- Providing GradScaler for loss scaling in float16 training
- Supporting multiple backends (cuda, cpu, xpu)
- Integrating with optimizers for gradient scaling

## Dependencies

**Inbound**: Training code
**Outbound**: `torch/autograd`, `torch/optim`

## Trade-offs

**Automatic dtype selection vs. manual**: Autocast automatically selects dtypes (float16 for compute kernels, float32 for reductions), trading explicitness for convenience.

## Runtime Implications

**Dtype override**: Within autocast context, operations use selected dtype (typically float16).

**Loss scaling**: GradScaler scales losses before backward to prevent gradient underflow in float16 training.

## Performance Implications

**2-3x speedup**: Float16 operations are typically 2-3x faster on GPUs with specialized hardware (Tensor Cores).

**Memory savings**: Float16 tensors use half memory compared to float32.

## Ownership Boundaries

- **AMP owns**: dtype selection and loss scaling logic
- **Operators own**: float16 implementations

## Verification Points

- `torch/amp/__init__.py` — Public interface
- `torch/amp/autocast.py` — Autocast implementation
- `torch/amp/grad_scaler.py` — GradScaler
