# Architecture Decision Record: torch/_functorch

## Architectural Role

`torch/_functorch` provides Python-level Functorch APIs: vmap (vectorization), grad (differentiation), and their compositions. It is the user-facing interface for functional transformations, enabling researchers to compose gradient and batching operations programmatically. Functorch enables advanced use cases like per-sample gradients and batch-level parallelism.

## Responsibilities

- Implementing vmap (vectorized map) for batched tensor transformations
- Implementing grad_and_value, vjp, jvp for differentiation strategies
- Supporting composition of transformations (grad ∘ vmap, etc.)
- Providing convenient APIs for common patterns (per-sample gradients via vmap)

## Dependencies

**Inbound**: User code, research frameworks
**Outbound**: `aten/src/ATen/functorch`, `torch/autograd`

## Trade-offs

**Dynamic dispatch in vmap**: Vmap dynamically determines batching rules per operation, adding overhead but enabling extensibility.

## Runtime Implications

**Batch dimension tracking**: Vmap tracks an additional dimension throughout computation.

**Composability**: Transformations can be nested (vmap(grad(...))

).

## Performance Implications

**Vmap overhead**: 5-15% slower than hand-coded batching for typical operations.

## Ownership Boundaries

- **Functorch owns**: transformation APIs
- **Operators own**: batching rules

## Verification Points

- `torch/_functorch/__init__.py` — Public interface
- `torch/_functorch/functional/` — Implementation
