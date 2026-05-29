# Architecture Decision Record: torch/distributed/fsdp

## Architectural Role

`torch/distributed/fsdp` (Fully Sharded Data Parallel) implements parameter sharding for memory-efficient large-model training. It automatically partitions model parameters across GPUs, gathering them only during computation. FSDP enables training of models larger than single-GPU memory, making it Runtime Critical for large-scale training.

## Responsibilities

- Implementing parameter sharding strategy (how to partition parameters across ranks)
- Managing parameter gathering/scattering during forward/backward
- Coordinating with NCCL for communication
- Implementing backward communication ordering for efficiency

## Dependencies

**Inbound**: Large-model training code
**Outbound**: `torch/distributed` for collectives, `torch/nn/modules` for module wrapping

## Trade-offs

**All-gather overhead during forward**: FSDP gathers parameters during forward, adding communication cost. This is offset by reduced parameter memory and communication efficiency.

**Overlapping communication with computation**: Advanced FSDP configurations overlap AllGather with backward computation, requiring careful gradient timing.

## Runtime Implications

**FSDP wrapper**: Models are wrapped with `FSDP()` to enable parameter sharding.

**Forward/backward synchronization**: Parameters are gathered during forward, scattered during backward.

## Performance Implications

**Communication overhead**: 20-50% for large models (offset by memory savings).

**Memory savings**: 4-8x memory reduction (per-GPU parameter memory divided by world size).

**Computation/communication overlap**: Can recover 20-30% of communication overhead via overlapping.

## Ownership Boundaries

- **FSDP owns**: parameter sharding and synchronization
- **ProcessGroup owns**: actual communication
- **Module owns**: computation

## Verification Points

- `torch/distributed/fsdp/fully_sharded_data_parallel.py` — FSDP implementation
- `torch/distributed/fsdp/_common_utils.py` — Sharding utilities
