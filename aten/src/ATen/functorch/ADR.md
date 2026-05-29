# Architecture Decision Record: aten/src/ATen/functorch

## Architectural Role

`aten/src/ATen/functorch` provides the C++ layer of Functorch, enabling vectorized transformations (vmap) and compositional functional operations. It is the bridge between Python-level Functorch APIs and the C++ kernel implementations, implementing batching rules and operator-specific vmap logic.

## Responsibilities

- Implementing vmap (vectorized map) for batched tensor transformations
- Batching rules for operators (how they behave under vmap)
- Composable functional transformations (grad, vmap, jit)
- Batch dimension tracking and inference

## Dependencies

**Inbound**: `torch/_functorch`, Python layer
**Outbound**: `aten/src/ATen/native`, dispatcher

## Trade-offs

**Batching rule per operator**: Each operator requires explicit batching rules, trading per-operator implementation for efficient vmap. The alternative (automatic batching) would be slower.

## Runtime Implications

**Batch dimension management**: Tracked separately from tensor shape to enable correct operator execution under vmap.

**Composability**: Vmap can be composed with grad and jit, enabling gradients of vectorized operations.

## Ownership Boundaries

- **Batching rules own**: transformation logic for specific operators

## Verification Points

- `aten/src/ATen/functorch/` — Implementation directory
