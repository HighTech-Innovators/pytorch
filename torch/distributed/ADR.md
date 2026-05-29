# Architecture Decision Record: torch/distributed

## Architectural Role

`torch/distributed` provides distributed training infrastructure, enabling models to be trained across multiple GPUs/TPUs and machines. It implements ProcessGroup (collective communication abstraction), DDP (Distributed Data Parallel), and hooks for custom gradient synchronization. This module is Runtime Critical for multi-GPU training at scale.

## Responsibilities

- Implementing ProcessGroup (abstraction for collective operations: allreduce, broadcast, reduce, gather, scatter)
- Implementing backends (NCCL for GPUs, Gloo for CPU, UCC, MPI)
- Providing DDP (Distributed Data Parallel) wrapper for modules
- Implementing communication hooks for custom synchronization patterns
- Managing rank/world-size assignment and process coordination
- Providing utilities for multi-GPU synchronization (barriers, broadcast)

## Dependencies

**Inbound** (what depends on torch/distributed):
- User training code
- FSDP, DDP submodules

**Outbound** (what torch/distributed depends on):
- Communication libraries (NCCL, Gloo, MPI)
- `torch/nn/modules` for DDP wrapping
- `torch/_C` for C++ bindings

## Trade-offs

**Backend abstraction vs. backend-specific optimization**: ProcessGroup abstracts different backends (NCCL, Gloo), trading per-backend tuning for portability.

**Eager gradient synchronization vs. bucketing**: DDP can synchronize gradients immediately (simple) or bucket them (more efficient). Bucketing reduces communication overhead but complicates deadlock debugging.

## Extension Boundaries

- **Custom backends**: New communication backends can implement the ProcessGroup interface.
- **Communication hooks**: Custom hooks enable domain-specific synchronization (e.g., gradient compression).

## Runtime Implications

**Process initialization**: `torch.distributed.init_process_group()` sets up communication backend and assigns ranks.

**Collective operations**: AllReduce, Broadcast, etc. are blocking (synchronous).

**Gradient synchronization**: DDP synchronizes gradients after backward, adding communication overhead (typically 10-30% of training time for multi-GPU).

## Performance Implications

**Communication overhead**: 10-30% of training time for multi-GPU (scales poorly beyond 4 GPUs on single machine).

**Bucketing efficiency**: Bucketing gradients reduces allreduce overhead by 20-30% but complicates debugging.

**Bandwidth utilization**: Modern interconnects (NVLink, Infiniband) are underutilized if collective operations are not carefully scheduled.

## Ownership Boundaries

- **ProcessGroup owns**: collective operation semantics
- **Backends own**: communication implementation
- **DDP owns**: gradient synchronization scheduling

## Verification Points

- `torch/distributed/__init__.py` — Public interface
- `torch/distributed/c10d/ProcessGroup.hpp` — ProcessGroup abstraction
- `torch/distributed/c10d/ProcessGroupNCCL.cpp` — NCCL backend
- `torch/distributed/parallel/DistributedDataParallel.py` — DDP implementation
