# Architecture Decision Record: torch/distributions

## Architectural Role

`torch/distributions` provides probability distribution implementations for stochastic modeling, variational inference, and reinforcement learning. It enables reparameterized sampling (differentiable through distributions) and probability computations, essential for generative models (VAEs), probabilistic inference, and policy gradient methods.

## Responsibilities

- Implementing distribution classes (Normal, Categorical, Uniform, Beta, Exponential, etc.)
- Providing sampling methods with reparameterized gradients (enabling gradient-through-sampling)
- Computing probability/density values (log_prob, prob) for likelihood evaluation
- Supporting distribution composition and transformation (TransformedDistribution)
- Integrating with torch.autograd for gradient computation through stochastic operations

## Dependencies

**Inbound** (what depends on torch/distributions):
- Generative models (VAEs, normalizing flows, diffusion models)
- Reinforcement learning (policy gradient methods, Q-learning)
- Probabilistic programming and Bayesian inference
- Research code using stochastic neural networks

**Outbound** (what torch/distributions depends on):
- `torch/autograd` for gradient computation
- `aten/src/ATen/native` for mathematical operations (log, exp, etc.)
- `c10/core` for tensor abstractions

## Trade-offs

**Reparameterized gradients vs. other gradient estimators**: The module implements reparameterized gradients (low-variance) rather than score function estimators (high-variance). This trades some flexibility for better optimization performance.

**Broadcasting support**: Distributions support batch and event shapes with automatic broadcasting, trading simplicity for flexibility in specifying batch dimensions.

## Extension Boundaries

- **Custom distributions**: Users can subclass `Distribution` to implement new distribution types.
- **Distribution transformations**: TransformedDistribution enables composing transformations (e.g., log-normal from normal).

## Runtime Implications

**Sampling**: Each call to `.sample()` draws new samples with reparameterized gradients enabling backpropagation.

**Probability computation**: `.log_prob()` computes log probability for use in loss functions or likelihood evaluation.

**Batch operations**: Distributions support batch dimensions for efficient sampling/evaluation of multiple distribution instances.

## Performance Implications

**Sampling overhead**: Reparameterized sampling adds 2-5% overhead compared to non-differentiable sampling but enables gradient computation.

**Probability computation**: Log probability is typically O(1) per sample, making it efficient for loss computation.

**Broadcasting efficiency**: Vectorized operations over batch dimensions provide 5-10x speedup compared to Python loops.

## Ownership Boundaries

- **Distribution owns**: sampling logic and probability computation
- **Autograd owns**: gradient computation through stochastic operations
- **Tensors own**: the underlying data

## Verification Points

- `torch/distributions/__init__.py` — Public API interface
- `torch/distributions/distribution.py` — Base Distribution class
- `torch/distributions/normal.py` — Example implementation (Normal distribution)
