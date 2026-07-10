# `torch/distributed`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/distributed` is the Python surface for multi-process collective communication. It exposes the process-group API, collective operations (allreduce, broadcast, gather, scatter, barrier), the rendezvous subsystem, and the distributed training strategies FSDP and DDP.

## Key Files

| File | Purpose |
|---|---|
| `torch/distributed/__init__.py` | Module entry; imports from `torch._C._distributed_c10d` when available; guards all APIs behind `is_available()` |
| `torch/distributed/distributed_c10d.py` | Core collective API: `init_process_group`, `all_reduce`, `broadcast`, `gather`, process-group lifecycle |
| `torch/distributed/device_mesh.py` | `DeviceMesh`, `init_device_mesh` — N-dimensional device topology for SPMD |
| `torch/distributed/fsdp/` | Fully Sharded Data Parallel: parameter sharding, optimizer state management |
| `torch/nn/parallel/distributed.py` | `DistributedDataParallel` (DDP) — gradient synchronization via `Reducer` |
| `torch/distributed/rendezvous.py` | Store-backed rendezvous: `register_rendezvous_handler`, `rendezvous` |

## Public Interface

`init_process_group()`, `destroy_process_group()`, `get_rank()`, `get_world_size()`, `all_reduce()`, `broadcast()`, `all_gather()`, `reduce_scatter()`, `barrier()`, `ProcessGroup`, `Backend`, `Store`, `TCPStore`, `FileStore`, `DeviceMesh`, `init_device_mesh()`, `rendezvous()`, `FullyShardedDataParallel`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._distributed_c10d` | depends-on | C++ process-group implementation, `Reducer`, `Work` objects |
| [torch/nn/parallel](torch/nn/parallel/ADR.md) | depended-on-by | DDP wraps `Module` and uses `Reducer` from this subsystem |
| [c10/core](c10/core/ADR.md) | depends-on | Tensor metadata and device types |

## Runtime Behaviour

`init_process_group()` in `distributed_c10d.py` constructs a `ProcessGroup` via the C++ `_distributed_c10d` extension; the concrete backend (NCCL, Gloo, MPI) is selected at this point. Collective operations return `Work` objects that represent asynchronous completions — callers must explicitly wait or the framework waits implicitly at synchronization points. The `is_available()` guard in `__init__.py` checks for `torch._C._c10d_init`; when `USE_DISTRIBUTED=0`, the module stubs out `ProcessGroup` and `GroupName` to satisfy import-time checks without any functional collectives.

## Performance Profile

Collective operations cross the Python↔C++ boundary once per call (into `torch._C._distributed_c10d`) and then block or return a `Work` handle. The dominant cost on CPU-only builds is inter-process data movement over shared memory or TCP, not kernel launch overhead. `Reducer` in DDP accumulates per-bucket gradients and fires all-reduce asynchronously during backward — bucket size controls communication granularity and overlaps compute with communication. When `USE_DISTRIBUTED=0`, the stubs add no runtime cost; any call path that reaches `is_available()` before attempting a collective incurs only a single attribute lookup.

## Design Rationale

The `is_available()` guard separates the build-time compile decision (`USE_DISTRIBUTED`) from runtime import behaviour, allowing the `torch.distributed` module to always exist as an importable name without raising `ImportError` in CPU-only or single-process deployments. The C++ `_distributed_c10d` bridge owns all backend-specific transport logic; the Python layer is policy-only (group management, rendezvous, tensor-list handling). `DeviceMesh` abstracts multi-dimensional process topologies so FSDP and tensor-parallel strategies share a unified rank-addressing scheme.
