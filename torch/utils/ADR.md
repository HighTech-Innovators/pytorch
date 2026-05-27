# Architecture Decision Record: torch.utils

## Architectural Role

The `torch.utils` subsystem provides **runtime utilities and infrastructure** that support training, data loading, and model persistence. These include:

1. **Data loading**: Efficiently batching and sampling datasets
2. **Checkpointing**: Saving and restoring model and optimizer state
3. **Monitoring**: Memory tracking and profiling hooks
4. **Serialization**: Saving/loading models and checkpoints

Key insight: torch.utils is **supporting infrastructure** — not core to training (torch.nn, torch.optim, torch.autograd handle that), but essential for practical workflows.

## Responsibilities

### What This Subsystem Owns

1. **Data Loading** (`data/`)
   - `DataLoader`: Batches and shuffles datasets with multiprocessing support
   - `Dataset` and `IterableDataset`: Base classes for custom datasets
   - `Sampler`: Defines iteration order over dataset
   - `DistributedSampler`: Partitions data across processes

2. **Checkpointing** (`checkpoint/`)
   - Save/load model state via `torch.save()` / `torch.load()`
   - Pickle-based serialization format
   - Device-agnostic checkpoints (save on GPU, load on CPU)

3. **Model Profiling and Monitoring** (`profiler.py`, in torch.profiler)
   - Hook infrastructure for model profiling
   - Memory tracking via weakrefs
   - Bottleneck identification

4. **Bundled Models and Datasets** (implicit)
   - Utilities for downloading and caching pretrained models
   - Dataset utilities (though many datasets in torchvision, not torch.utils)

## Key Implementation Files

| File | Purpose |
|---|---|
| `data/dataloader.py` | DataLoader main class |
| `data/dataset.py` | Dataset base classes |
| `data/sampler.py` | Sampler implementations |
| `checkpoint.py` | Model save/load utilities |

---

# Architecture Decision Record: torch.backends

## Architectural Role

The `torch.backends` subsystem provides **backend-specific configuration and utilities** for device acceleration (CUDA, MPS, etc.). It enables:

1. **Backend selection**: Choose between available backends
2. **Backend configuration**: Enable/disable optimizations, set precision modes
3. **Device queries**: Query backend capabilities

Key insight: torch.backends is **thin wrapper over device-specific settings** — most device logic is in c10/ATen, but this module provides the user-facing configuration API.

## Responsibilities

### What This Subsystem Owns

1. **CUDA Configuration** (`backends/cuda/`)
   - `torch.backends.cuda.is_available()`: Check if CUDA available
   - `torch.backends.cuda.enabled`: Enable/disable CUDA
   - Matmul precision, cuDNN configuration

2. **cuDNN Configuration** (`backends/cudnn/`)
   - Benchmark mode: auto-tune convolution algorithms
   - Deterministic mode: reproducible results
   - Enabled/disabled flags

3. **Other Backend Configuration** (`backends/mps/`, `backends/xpu/`)
   - MPS (Apple Metal Performance Shaders)
   - XPU (Intel extensions)

## Key Implementation Files

| File | Purpose |
|---|---|
| `backends/cuda/__init__.py` | CUDA backend configuration |
| `backends/cudnn/__init__.py` | cuDNN configuration |

Last Verified: 2026-05-27
