# `torch/nn/parallel`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/nn/parallel` provides data-parallel training wrappers: `DistributedDataParallel` (DDP), which replicates a module across processes and synchronizes gradients via collective all-reduce, and the legacy single-process, multi-GPU `DataParallel`. These wrap an `nn.Module` to distribute the forward/backward across devices or ranks. **In this CPU-only, `USE_DISTRIBUTED=0` deployment these wrappers are import-visible but not functional at runtime.**

## Key Files

| File | Purpose |
|---|---|
| `torch/nn/parallel/distributed.py` | `DistributedDataParallel` (2666 lines): gradient bucketing, all-reduce hooks |
| `torch/nn/parallel/data_parallel.py` | `DataParallel`: single-process multi-GPU replication |
| `torch/nn/parallel/replicate.py` | Replicates a module's parameters/buffers across devices |
| `torch/nn/parallel/scatter_gather.py` | Scatters inputs to replicas, gathers outputs |
| `torch/nn/parallel/parallel_apply.py` | Runs replica forwards concurrently on threads |
| `torch/nn/parallel/comm.py` | Broadcast/reduce primitives across devices |

## Public Interface

`torch.nn.parallel.DistributedDataParallel` (`nn.parallel.DDP`), `torch.nn.DataParallel`, `replicate()`, `scatter()`, `gather()`, `parallel_apply()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | Wraps an `nn.Module`; reuses parameter/buffer traversal |
| [torch/distributed](torch/distributed/ADR.md) | depends-on | DDP uses process groups and all-reduce collectives |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Registers backward hooks to trigger gradient synchronization |
| `torch._C._distributed_c10d` | depends-on | C++ collective backend (absent when `USE_DISTRIBUTED=0`) |
| Training code | depended-on-by | User training loops wrap the model in DDP |

## Runtime Behaviour

When functional, `DistributedDataParallel` broadcasts the initial module state so all ranks start identical, then groups parameters into gradient buckets. During backward, autograd hooks fire as each parameter's gradient becomes ready and enqueue an asynchronous all-reduce for its bucket, overlapping communication with continued backward computation; `.grad` holds the averaged gradient once all-reduce completes. `DataParallel` instead replicates the module per forward call, scatters the input batch, runs replicas via `parallel_apply`, and gathers outputs on the primary device. In this deployment `torch.distributed.is_available()` is false, so constructing DDP raises rather than running these paths.

## Performance Profile

DDP's core optimization is overlapping gradient all-reduce with backward compute via bucketing — communication is network-bandwidth bound on multi-node setups, and bucket size trades latency against overlap. `DataParallel` is limited by the single-process GIL and by scatter/gather copies to and from the primary device, which is why DDP is preferred. Neither mechanism contributes runtime cost here because the distributed backend is disabled; the note matters only for understanding why importing these classes succeeds while using them fails.

## Design Rationale

DDP exists because scaling training across processes/nodes requires synchronized gradients with minimal stalls; bucketed, hook-triggered async all-reduce is the design that keeps GPUs busy during synchronization. It supersedes `DataParallel`, whose single-process threading model is GIL- and copy-bound. This subsystem is retained in a CPU-only build for API compatibility, but the conditional `torch.distributed.is_available()` gating (see the distributed ADR) means code must check availability or it fails at call time rather than import time — a known health concern for this configuration.
