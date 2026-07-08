# `torch/utils`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/utils` collects the framework-level Python utilities that do not belong to a single tensor operator namespace. It includes data loading, checkpoint/recompute, C++ extension compilation, benchmarking, hooks, pytree helpers, tensorboard support, model mobile utilities, environment collection, debug modes, serialization configuration, and visualization helpers. It complements book Chapter 06 (`book/06-python-bindings.md`) by providing the Python-side infrastructure around tensors and extensions, and it complements Chapter 12 (`book/12-observability.md`) through benchmark, traceback, and visualization utilities.

## Key Files

| File | Purpose |
|---|---|
| `data/dataloader.py` | Python `DataLoader`, dataset-kind dispatch, worker setup, distributed sharding seed sharing, multiprocessing iterators, and pin-memory integration |
| `data/dataset.py` | `Dataset`, `IterableDataset`, `TensorDataset`, `StackDataset`, `ConcatDataset`, and subset abstractions |
| `data/sampler.py` | Sequential, random, subset-random, weighted-random, batch, and distributed sampling primitives |
| `checkpoint.py` | Activation checkpointing, RNG state preservation, selective checkpoint policies, debug controls, and recomputation helpers |
| `cpp_extension.py` | C++/CUDA/SYCL extension build helpers, compiler discovery, ABI checks, include/library paths, and JIT loading entry points |
| `benchmark/utils/timer.py` | PyTorch-aware benchmark timer with accelerator synchronization, C++ timing support, replicates, and metadata grouping |
| `hooks.py` | `RemovableHandle`, unserializable-hook markers, tensor hook serialization warnings, and module backward-hook plumbing |
| `_pytree.py` | Tree flattening, unflattening, mapping, and registration utilities used across compiler and API code |
| `_python_dispatch.py` | Python `TorchDispatchMode` support and dispatch-mode utilities |
| `tensorboard/writer.py` | Summary writer integration for TensorBoard logging |

## Public Interface

Public APIs include `torch.utils.data.DataLoader`, dataset and sampler classes, `torch.utils.checkpoint.checkpoint`, `checkpoint_sequential`, selective activation checkpoint contexts, `torch.utils.cpp_extension.load`, `load_inline`, `CppExtension`, `CUDAExtension`, `BuildExtension`, `include_paths`, `library_paths`, `torch.utils.benchmark.Timer`, `Compare`, `Fuzzer`, `torch.utils.hooks.RemovableHandle`, tensorboard writers, environment collection helpers, DLPack helpers, mobile optimizer helpers, and pytree registration utilities. Many subpackages expose their own `__all__` lists while `torch.utils` serves as a stable namespace for advanced application infrastructure.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | related | Utilities operate on devices, tensors, storage, and dispatch concepts exposed through core tensor objects |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | related | Data, checkpoint, benchmark, and extension helpers ultimately exercise ATen tensors and operators |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Checkpointing, hooks, saved tensors, and backward-hook support integrate directly with autograd behavior |
| [torch/cuda](torch/cuda/ADR.md) | related | DataLoader pin memory, checkpoint RNG preservation, benchmark synchronization, and C++ extension CUDA paths use CUDA APIs when present |
| [torch/distributed](torch/distributed/ADR.md) | related | DataLoader distributed sharding and distributed samplers coordinate dataset partitioning across process groups |
| [torch/fx](torch/fx/ADR.md) | related | Checkpoint debug paths and pytree utilities interact with FX traceback and compiler infrastructure |

## Runtime Behaviour

`DataLoader` chooses map-style or iterable-style fetching, configures samplers and batch samplers, optionally starts multiprocessing workers, seeds workers, shards DataPipes across distributed ranks and workers, prefetches batches, and optionally pins memory before yielding results. `checkpoint` saves selected RNG states, runs the forward function with controlled autograd recording, and recomputes activations during backward to reduce saved tensor memory. `cpp_extension` discovers compilers and CUDA/ROCM/SYCL homes, checks ABI compatibility, constructs setuptools or Ninja builds, coordinates concurrent builds with `FileBaton`, and loads built extension modules. `Timer` synchronizes accelerators before timing, supports Python and C++ snippets, repeats measurements, and records metadata for comparison tables. `RemovableHandle` stores weak references to hook dictionaries and removes matching entries when users call `remove` or exit its context.

## Performance Profile

`DataLoader` improves input throughput by overlapping dataset work with training through multiprocessing workers, prefetching, persistent workers, and pinned-memory transfer, while in-order delivery and Python collation can become bottlenecks for small batches. Activation checkpointing trades extra compute for lower memory by discarding intermediates and recomputing them during backward; RNG preservation adds overhead proportional to the number of involved devices. `cpp_extension` front-loads compiler detection and build cost, then reuses versioning and file locks to avoid unnecessary or conflicting rebuilds. `Timer` synchronizes accelerators to measure asynchronous kernels correctly and uses replicate-focused measurement so medians resist run-to-run noise.

## Design Rationale

These utilities live together because they support application construction, debugging, extension, and measurement rather than one dispatcher subsystem. Data loading stays in Python because datasets, transforms, and collation are user-defined Python objects, while worker processes and pin-memory threads isolate the expensive parts. Checkpointing belongs in utilities because it is a policy layer over autograd rather than a new tensor operation. C++ extension helpers give users a supported path from Python packages to custom native operators without requiring them to duplicate PyTorch's compiler, include-path, ABI, and CUDA-discovery logic.
