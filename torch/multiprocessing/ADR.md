# `torch/multiprocessing`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/multiprocessing` wraps Python's `multiprocessing` module with tensor-aware serialization. It registers reducers that move CPU storages into shared memory, transfer CUDA tensors through IPC handles, and provide spawn helpers that propagate child-process failures with original tracebacks.

## Key Files

| File | Purpose |
|---|---|
| `torch/multiprocessing/__init__.py` | Imports all names from `multiprocessing`, initializes C++ multiprocessing helpers, exposes sharing-strategy APIs, and calls `init_reductions()` |
| `torch/multiprocessing/reductions.py` | Registers tensor, storage, `Parameter`, and CUDA event reducers and rebuild functions for shared memory and CUDA IPC |
| `torch/multiprocessing/spawn.py` | Implements `ProcessContext`, `SpawnContext`, `start_processes`, `spawn`, child traceback files, sentinel waiting, and failure cleanup |
| `torch/multiprocessing/queue.py` | Wraps multiprocessing connections with `ForkingPickler` so queues use PyTorch reducers |
| `torch/multiprocessing/pool.py` | Overrides `Pool` queues with PyTorch `SimpleQueue` and runs `gc.collect()` after worker shutdown |

## Public Interface

| Symbol | Description |
|---|---|
| `set_sharing_strategy`, `get_sharing_strategy`, `get_all_sharing_strategies` | Configure CPU storage sharing as `file_descriptor` or `file_system` depending on platform support |
| `spawn(fn, args, nprocs, join, daemon, start_method)` | Starts `nprocs` child processes with `spawn` semantics and forwards child exceptions to the parent |
| `start_processes` | Generalized process launcher that supports `spawn`, `fork`, and opt-in parallel `forkserver` starts through `TORCH_MP_PARALLEL_START` |
| `ProcessContext` / `SpawnContext` | Track child processes, expose `pids()`, and implement `join()` with termination and traceback handling |
| `Queue`, `SimpleQueue`, `Pool` | Multiprocessing containers that serialize tensors through `ForkingPickler` and PyTorch reducers |
| `init_reductions` | Registers reducers for `torch._storage_classes`, `torch._tensor_classes`, `torch.Tensor`, `torch.nn.parameter.Parameter`, and `torch.cuda.Event` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses tensor classes, storage classes, `TypedStorage`, `UntypedStorage`, `Parameter`, sparse layouts, nested tensors, and CUDA APIs |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Calls `torch._C._multiprocessing_init`, `_set_thread_name`, and `_get_thread_name` for C++ process/thread integration |
| [c10/core](c10/core/ADR.md) | depends-on | Shares and rebuilds underlying storage objects that wrap c10 allocation and device metadata |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | Uses CUDA IPC handles, CUDA event IPC, and caching-allocator metadata for CUDA tensor sharing |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Rejects serialization of non-leaf tensors that require grad and rebuilds `torch.nn.parameter.Parameter` with `requires_grad` |

## Runtime Behaviour

Importing `torch.multiprocessing` calls `torch._C._multiprocessing_init()`, imports Python `multiprocessing` symbols, chooses a default CPU sharing strategy, and registers all PyTorch reducers. When a tensor crosses a queue or pipe, `reduce_tensor` rejects non-leaf tensors requiring grad, handles nested and sparse layouts, and returns rebuild functions plus storage metadata. `spawn.start_processes` creates one traceback file per child, waits on process sentinels, terminates surviving children after the first failure, and raises `ProcessRaisedException` or `ProcessExitedException` in the parent.

## Performance Profile

CPU tensor transfer avoids data copies after storage sharing: `reduce_storage` uses `_share_fd_cpu_()` for `file_descriptor` mode and `_share_filename_cpu_()` for `file_system` mode, then caches handles in `SharedCache`. CUDA tensor transfer sends an IPC handle for the entire underlying `cudaMalloc` allocation because the caching allocator can place the tensor storage inside a larger allocation; `rebuild_cuda_tensor` reuses `shared_cache` entries and releases producer IPC counters when a storage is already cached. `start_processes` starts workers sequentially by default and starts `forkserver` workers in parallel only when `TORCH_MP_PARALLEL_START=1`.

## Design Rationale

The module preserves the standard `multiprocessing` API by importing `multiprocessing.__all__` and adding PyTorch-specific behavior through reducers and queue wrappers. This lets users change `import multiprocessing` to `import torch.multiprocessing` without rewriting process orchestration code. The reducer design keeps tensor payloads in shared storage whenever possible while preserving autograd safety boundaries and CUDA IPC lifetime rules.
