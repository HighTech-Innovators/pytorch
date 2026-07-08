# `torch/multiprocessing`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/multiprocessing` owns PyTorch's multiprocessing compatibility layer. It preserves the standard `multiprocessing` API surface while adding tensor-aware reducers, shared-memory transfer, CUDA IPC rebuilding, and coordinated worker startup and failure handling.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the stdlib multiprocessing API, configures sharing strategies, calls `_multiprocessing_init`, and registers PyTorch reducers |
| `spawn.py` | Implements `ProcessContext`, `spawn`, `start_processes`, and failure propagation across worker groups |
| `reductions.py` | Rebuilds and reduces CPU, CUDA, nested, and sparse tensor storages for inter-process transfer |
| `queue.py` | Defines PyTorch-aware queue wrappers that use the registered reducers |

## Public Interface

The package exports stdlib multiprocessing symbols plus `set_sharing_strategy`, `get_sharing_strategy`, `get_all_sharing_strategies`, `spawn`, `start_processes`, `ProcessContext`, `SpawnContext`, `ProcessRaisedException`, and `ProcessExitedException`. Tensor-transfer entry points include the reducer registration done by `init_reductions()` and rebuild helpers such as `rebuild_tensor`, `rebuild_cuda_tensor`, `rebuild_nested_tensor`, and `rebuild_sparse_compressed_tensor`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nested](torch/nested/ADR.md) | depends-on | `reductions.py` contains `rebuild_nested_tensor` and `reduce_nested_tensor` support for nested tensors |
| [torch/sparse](torch/sparse/ADR.md) | depends-on | `reductions.py` serializes and rebuilds sparse COO and compressed sparse tensors |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | distributed launch and training code uses `spawn` and `start_processes` to manage rank-local workers |

## Runtime Behaviour

Importing `torch.multiprocessing` first re-exports `multiprocessing.__all__`, then calls `torch._C._multiprocessing_init()` and `init_reductions()` so queues and pipes understand PyTorch tensors. `spawn._wrap()` runs the user function in each child, catches exceptions, pickles the traceback into an error file, and exits with status `1` so the parent can reconstruct the failure. `ProcessContext.join()` waits on process sentinels, terminates surviving peers when any process exits abnormally, and raises either `ProcessExitedException` or `ProcessRaisedException` with the recorded details. `reductions.py` rebuilds CPU tensors from shared storage metadata, rebuilds CUDA tensors through `_new_shared_cuda()` and IPC refcount handles, and preserves `requires_grad` correctly for `torch.nn.parameter.Parameter`.

## Performance Profile

The primary optimization is data movement avoidance: once a tensor's storage is in shared memory or CUDA IPC space, child processes can rebuild views without copying payload bytes. `SharedCache` retains weak references to opened storages, which prevents repeated CUDA handle opens and amortizes IPC setup cost across receives. `start_processes()` can parallelize worker startup for `forkserver` when `TORCH_MP_PARALLEL_START=1`, trading a small thread-pool cost for lower wall-clock launch time. The failure-handling path is intentionally expensive because it waits, terminates, and may kill remaining workers to return a deterministic first error.

## Design Rationale

PyTorch extends stdlib multiprocessing instead of replacing it so existing multiprocessing code can switch imports with minimal friction. Reducer registration and structured spawn management live in one package because safe multi-process tensor execution needs both efficient transport and predictable worker lifecycle control.
