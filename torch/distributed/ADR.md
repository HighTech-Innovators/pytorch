# `torch/distributed`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/distributed` is the Python distributed-training layer. It exposes c10d collectives, process-group management, DDP communication hooks, FSDP, RPC, distributed checkpointing, elastic launch, device meshes, tensor parallel building blocks, and distributed optimizers. It uses the Python binding pattern described in book Chapter 06 (`book/06-python-bindings.md`): Python functions validate arguments and manage process-global state, then call native classes and functions from `torch._C._distributed_c10d` and `torch._C._distributed_rpc`.

## Key Files

| File | Purpose |
|---|---|
| `distributed_c10d.py` | Main collective API, process-group registry, backend discovery, rendezvous integration, and wrappers for native c10d classes |
| `rendezvous.py` | Rendezvous handler registry and URL-based initialization helpers |
| `fsdp/fully_sharded_data_parallel.py` | `FullyShardedDataParallel` module wrapper and public FSDP orchestration API |
| `fsdp/_runtime_utils.py` | FSDP pre-forward, post-forward, unshard, reshard, and lazy runtime helpers |
| `rpc/api.py` | Python RPC APIs for `rpc_sync`, `rpc_async`, `remote`, `RRef`, shutdown, worker info, and initialization checks |
| `algorithms/ddp_comm_hooks/default_hooks.py` | Default DDP allreduce and FP16/BF16 compression hooks built on futures and async collectives |
| `checkpoint/state_dict_saver.py` | Distributed checkpoint save orchestration and state-dict planning entry points |
| `device_mesh.py` | Device mesh abstraction for mapping logical parallel dimensions onto process groups |
| `elastic/rendezvous/dynamic_rendezvous.py` | Elastic membership and rendezvous coordination for fault-tolerant jobs |
| `run.py` | `torchrun` command implementation for launching distributed workers |

## Public Interface

The package exports `init_process_group`, `destroy_process_group`, `new_group`, `get_rank`, `get_world_size`, collective functions such as `all_reduce`, `broadcast`, `reduce_scatter_tensor`, `all_gather_into_tensor`, point-to-point `send`/`recv`, `Work`, `Store`, `ProcessGroup`, and `ReduceOp`. Higher-level APIs expose `FullyShardedDataParallel`, DDP communication hooks, RPC functions and `RRef`, distributed checkpoint save/load, elastic launch utilities, `DeviceMesh`, distributed tensor helpers, and distributed optimizers. Most collective APIs accept optional process groups and return native `Work` objects when `async_op=True`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/distributed](torch/csrc/distributed/ADR.md) | depends-on | Wraps native c10d, RPC, process-group, work, store, reducer, and distributed-autograd classes |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | DDP, FSDP, RPC autograd, and checkpointing integrate with gradient recording and backward hooks |
| [torch/nn](torch/nn/ADR.md) | depends-on | FSDP and DDP wrap `nn.Module` instances and rewrite or shard their parameters |
| [torch/optim](torch/optim/ADR.md) | depends-on | Distributed optimizers, optimizer-overlap hooks, and checkpoint planners operate on optimizer state |
| [torch/_dynamo](torch/_dynamo/ADR.md) | related | Functional collectives and FSDP Dynamo annotations make distributed code visible to `torch.compile` |

## Runtime Behaviour

`init_process_group` resolves a rendezvous method, creates or receives a store, constructs a native process group for the selected backend, and registers it in Python process-global group state. Collective wrappers normalize tensor lists, process groups, options, and async flags, then call the native `ProcessGroup` method and optionally return a `Work` object or block for completion. FSDP wraps modules, flattens or tracks parameters, unshards parameters before forward/backward computation, reshares them afterward, and wires state-dict and optimizer-state transformations around the sharded representation. RPC APIs require an initialized native agent, pickle or encode Python/TorchScript calls, return futures or RRefs, and coordinate shutdown and RRef cleanup.

## Performance Profile

The package keeps collective execution in C++ and uses asynchronous `Work` or future objects so Python orchestration does not block communication overlap unless the caller asks to wait. DDP communication hooks reduce bandwidth or overlap optimizer work by operating directly on gradient buckets and chaining futures. FSDP trades extra all-gather and reduce-scatter communication for lower model-state memory by sharding parameters, gradients, and optimizer state across ranks. Distributed checkpointing spends most time in storage I/O, planning, and tensor movement, so it separates metadata planning from storage writers and offers async executors.

## Design Rationale

The Python layer owns usability, argument normalization, process-global registries, and integration with `nn.Module`, while the C++ layer owns communication hot paths. This split keeps user APIs Pythonic without putting serialization, network progress, and CUDA stream coordination in Python. FSDP lives here because sharding changes module state, optimizer state, and communication together; a single package can coordinate those concerns. RPC, elastic launch, checkpoint, and device mesh remain under one namespace because distributed applications need them together even though their runtime mechanisms differ.
