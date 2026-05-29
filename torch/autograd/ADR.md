# Architecture Decision Record: torch/autograd

## Architectural Role

`torch/autograd` provides the Python-level interface to PyTorch's automatic differentiation system, exposing the computation graph, gradient computation, and custom autograd functions to users. It is Runtime Critical for training; nearly all training code uses this module either explicitly (custom `autograd.Function`) or implicitly (backward passes).

## Responsibilities

- Implementing `torch.autograd.Function` for user-defined differentiable operations (forward/backward methods)
- Exposing `backward()` interface at the tensor level
- Providing context managers like `no_grad()` and `set_grad_enabled()`
- Implementing higher-order derivatives (gradients of gradients) via graph retracing
- Providing anomaly detection for backward pass debugging

## Dependencies

**Inbound** (what depends on torch/autograd):
- Training loops (calling `.backward()`)
- Custom gradient logic (subclassing `autograd.Function`)
- Compiler integration (torch.compile needs autograd information)

**Outbound** (what torch/autograd depends on):
- `torch/csrc/autograd` for C++ engine implementation
- `torch/_C` for performance-critical operations

## Trade-offs

**Python-level gradient tracking**: Overhead of Python function calls in the hot path is amortized by the C++ engine. The alternative (pure Python engine) would be simpler but too slow for production.

**autograd.Function inheritance for custom gradients**: Users define custom operations by subclassing `autograd.Function` and implementing `forward` and `backward`. This is flexible but requires understanding the autograd graph model.

## Extension Boundaries

- **Custom autograd.Function**: Users can define operations with custom backward by subclassing and implementing `forward` and `backward` methods.
- **Custom gradient implementations**: New operations register their backward via `autograd.Function`.

## Runtime Implications

**Gradient accumulation**: Gradients are accumulated on leaf tensors during the backward pass. Users typically zero gradients before each backward (`optimizer.zero_grad()`).

**Graph retracing for higher-order derivatives**: Computing gradients of gradients (e.g., Hessian-vector products) requires retracing the computation graph, which adds significant overhead.

**Anomaly detection**: Enabling anomaly detection (via `detect_anomaly()`) tracks forward operations for improved error messages but adds overhead.

## Performance Implications

**Python overhead**: The Python interface adds 1-2% overhead compared to direct C++ calls, but this is negligible compared to kernel execution time.

**Higher-order derivative overhead**: Computing Hessians or Jacobians is 10-100x slower than single backward pass, as the graph must be retracted.

## Ownership Boundaries

- **autograd.Function owns**: user-defined forward and backward implementations
- **Engine (torch/csrc/autograd) owns**: graph execution and gradient accumulation

## Verification Points

- `torch/autograd/__init__.py` — Public interface
- `torch/autograd/function.py` — autograd.Function base class
- `torch/autograd/grad_mode.py` — Gradient enable/disable contexts
