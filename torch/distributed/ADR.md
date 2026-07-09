# `torch/distributed`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/distributed` provides distributed training infrastructure: the collective communication library (c10d), `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP), `DeviceMesh`, RPC, and the functional collectives API. It coordinates multi-process and multi-node tensor computation over NCCL, Gloo, and MPI backends.

## Key Files

| File | Purpose |
|---|---|
| `distributed_c10d.py` | Core collective API: `init_process_group`, `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, `barrier`; wraps `torch._C._distributed_c10d.ProcessGroup` |
| `__init__.py` | Package entry: `is_available()` checks `torch._C._c10d_init`; imports all public symbols from `distributed_c10d` |
| `_functional_collectives.py` | Functional (non-in-place) collective wrappers compatible with `torch.compile`; `all_reduce_coalesced`, `all_gather_into_tensor` |
| `fsdp/_flat_param.py` | `FlatParameter` — flattened view of a set of parameters; used by FSDP for all-gather and reduce-scatter of parameter shards |
| `fsdp/_init_utils.py` | FSDP initialization: sharding strategy, mixed precision config, `DeviceMesh` wiring |
| `fsdp/_fully_shard/` | New FSDP2 implementation with composable sharding |
| `_composable/` | Composable distributed APIs: `replicate`, `shard` decorators |
| `_tensor/` | `DTensor` — distributed tensor with `DeviceMesh` placement annotations |
| `_mesh_layout.py` / `_sharding_spec/` | `DeviceMesh` — N-D mesh of device ranks; placement types (`Shard`, `Replicate`, `Partial`) |
| `rendezvous.py` | Rendezvous backends: `env://`, `file://`, `tcp://` — used by `init_process_group` to discover peers |
| `rpc/` | RPC framework: `rpc_init`, `remote`, `rpc_sync`, `rpc_async`, `rpc_backend_options` |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.distributed.init_process_group(backend, ...)` | Initialises c10d process group; backend is `"nccl"`, `"gloo"`, or `"mpi"` |
| `torch.distributed.all_reduce(tensor, op)` | In-place all-reduce across all ranks in the process group |
| `torch.distributed.ProcessGroup` | C++ object (`torch._C._distributed_c10d.ProcessGroup`) wrapping the backend collective implementation |
| `torch.distributed.DistributedDataParallel` | Module wrapper that hooks `backward()` to all-reduce gradients; `find_unused_parameters` option |
| `torch.distributed.fsdp.FullyShardedDataParallel` | Shards parameters and optimizer states across ranks; all-gathers before forward, reduce-scatters after backward |
| `torch.distributed.DeviceMesh` | N-D process group layout; `mesh["dp"]` selects a sub-mesh |
| `torch.distributed.DTensor` | Tensor with `DeviceMesh`-aware placement; operations produce new `DTensor`s |
| `torch.distributed.rpc.init_rpc(name, ...)` | Initialises the RPC agent for cross-process function calls |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._distributed_c10d` | depends-on | C++ `ProcessGroup` and collective implementations (NCCL, Gloo); accessed via `torch._C._c10d_init` |
| [torch/nn](torch/nn/ADR.md) | depends-on | `DistributedDataParallel` wraps an `nn.Module`; hooks its backward pass |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | DDP registers backward hooks on gradient tensors via `Variable._execution_engine.queue_callback` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Collective output tensors are ATen tensors; `all_gather_into_tensor` writes into pre-allocated ATen buffers |
| [c10/core](c10/core/ADR.md) | depends-on | `c10::Device`, `c10::Stream` used to associate collectives with CUDA streams |
| NCCL / Gloo / MPI (external) | depends-on | Collective communication backends; NCCL for GPU-GPU transfers |

## Runtime Behaviour

`init_process_group` calls `_DistributedBackendOptions` to configure the backend and creates a `ProcessGroup` C++ object. All collective operations (e.g., `all_reduce`) call into the C++ `ProcessGroup` API via `torch._C._distributed_c10d`; they enqueue work objects that are submitted to the backend's internal thread pool and return a `Work` handle. `DDP`'s backward hook fires after each gradient bucket's backward pass completes; it calls `all_reduce` on the bucket and synchronises when all buckets are reduced. FSDP's `FlatParameter` all-gather is triggered in the `forward` pre-hook and reduce-scatter in the post-backward hook; parameter memory is freed immediately after the forward pre-hook to minimise peak GPU memory.

## Performance Profile

- **Allocation sites**: DDP pre-allocates gradient buckets (default size 25 MB) at construction; FSDP pre-allocates flat-parameter buffers per shard. Per-iteration allocation is minimal once buckets are initialized.
- **Synchronization costs**: all-reduce via NCCL is asynchronous (returns a `Work` handle); `Work.wait()` blocks until the kernel completes. Overlapping communication with computation (gradient compression, bucket pipelining in DDP) reduces this blocking time. FSDP all-gathers block the forward pass; `prefetch_next_fsdp_module` reduces this by pre-fetching the next layer's parameters while the current layer runs.
- **Data movement**: every all-reduce moves gradient data across NVLink (intra-node) or InfiniBand (inter-node). FSDP reduce-scatter halves per-rank communication volume compared to DDP's all-reduce for large models. DTensor operations perform lazy communication, collecting collectives until a `.full_tensor()` call forces materialisation.
- **Redundant or repeated work**: DDP's `find_unused_parameters=True` adds a traversal of the autograd graph after each forward to detect unused parameters; this adds overhead proportional to model size and is disabled by default.

## Design Rationale

The separation of `distributed_c10d` (collective primitives) from `DistributedDataParallel` (module wrapper) allows the collective API to be used independently of the training abstraction. FSDP uses `FlatParameter` — a single contiguous view over multiple parameters — because NCCL performs best with large contiguous buffers; flattening eliminates per-parameter launch overhead. `DeviceMesh` is a first-class type rather than a simple rank list so that N-D parallelism (data-parallel + tensor-parallel + pipeline-parallel) can be expressed without reimplementing rank arithmetic in every user-defined distributed program.
