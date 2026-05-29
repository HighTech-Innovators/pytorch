# Architecture Decision Record: torch/multiprocessing

## Architectural Role

`torch/multiprocessing` provides multiprocessing utilities for distributed training and parallel data loading. It wraps Python's standard `multiprocessing` module with PyTorch-specific features: tensor sharing across processes (via shared memory), CUDA context management in worker processes, and process group initialization helpers for distributed training.

## Responsibilities

- Implementing `spawn()` for process launching with proper CUDA context initialization
- Providing tensor sharing across processes via shared memory (avoiding serialization overhead)
- Managing process lifecycle and error propagation from worker processes
- Implementing process group initialization helpers for distributed training
- Providing cross-platform support (Linux, macOS, Windows with different multiprocessing start methods)

## Dependencies

**Inbound** (what depends on torch/multiprocessing):
- `torch/utils/data/DataLoader` for multi-worker data loading
- Distributed training scripts using `torch.distributed.launch`
- Research code using parallel processing

**Outbound** (what torch/multiprocessing depends on):
- Python's `multiprocessing` module
- System IPC (shared memory, sockets)
- `torch/distributed` for process group initialization

## Trade-offs

**Shared memory tensor transfer vs. serialization**: Tensors are shared across processes via shared memory (fast) rather than serialized (slow). This trades complexity for efficiency.

**Automatic spawn vs. manual process creation**: `multiprocessing.spawn()` automatically handles CUDA context setup, trading some flexibility for ease of use.

## Extension Boundaries

- **Custom tensor communication**: Users can implement custom tensor sharing strategies.
- **Custom process launching**: Advanced users can implement alternative spawn strategies.

## Runtime Implications

**Process creation**: `spawn()` creates worker processes, each with independent CUDA context and memory.

**Tensor sharing**: Tensors passed to worker processes are shared via shared memory; modifications in workers are visible to parent (care needed to avoid data races).

**Worker errors**: Exceptions in worker processes are propagated to the parent process.

## Performance Implications

**Spawn overhead**: Process creation adds 100-500ms per worker, amortized across worker lifetime.

**Tensor sharing efficiency**: Shared memory transfer is fast (~1GB/s), avoiding serialization bottleneck of standard pickle.

**DataLoader efficiency**: Multi-worker data loading provides 2-5x throughput improvement for I/O-bound datasets.

## Ownership Boundaries

- **torch.multiprocessing owns**: process launching and tensor sharing
- **Worker processes own**: their own memory and CUDA context
- **Tensors own**: underlying data (shared via memory mapping)

## Verification Points

- `torch/multiprocessing/__init__.py` — Public API interface
- `torch/multiprocessing/spawn.py` — Process spawning
- `torch/multiprocessing/queue.py` — Cross-process communication
