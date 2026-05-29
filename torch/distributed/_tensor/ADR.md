# Architecture Decision Record: torch/distributed/_tensor

## Architectural Role

`torch/distributed/_tensor` (DTensor) provides the Distributed Tensor abstraction, enabling automatic tensor sharding across devices. DTensor abstracts device placement and sharding strategy, allowing users to write single-device code that automatically scales to multi-device. This is an advanced distributed training feature for large-scale training.

## Responsibilities

- Implementing DTensor (tensor with sharding metadata)
- Implementing sharding specs (data sharding, model sharding, etc.)
- Providing automatic communication insertion for cross-shard operations
- Supporting DTensor-to-tensor conversion and vice versa

## Dependencies

**Inbound**: Advanced distributed training code
**Outbound**: `torch/distributed` for collective operations

## Trade-offs

**Automatic sharding vs. explicit placement**: DTensor automatically inserts communication for cross-shard operations, trading fine-grained control for ease of use.

## Runtime Implications

**Sharding metadata**: Each DTensor carries sharding information used to determine communication patterns.

**Communication insertion**: Operations across shard boundaries automatically insert AllGather, Reduce, or Scatter.

## Performance Implications

**Communication overhead**: Similar to DDP but potentially higher due to per-operation decisions.

**Overhead of abstraction**: Sharding metadata checking adds small per-operation overhead.

## Ownership Boundaries

- **DTensor owns**: sharding metadata
- **Collective ops own**: actual communication

## Verification Points

- `torch/distributed/_tensor/__init__.py` — DTensor implementation
