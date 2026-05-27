# Architecture Decision Record: torch.distributed

## Architectural Role

The `torch.distributed` subsystem enables **multi-process and multi-machine training** by providing:

1. **Process group abstraction** — logical grouping of processes for collective operations
2. **Communication backends** (NCCL, Gloo, MPI) — efficient point-to-point and collective communication
3. **Collective operations** (allreduce, broadcast, scatter, gather) — synchronized tensor operations across processes
4. **High-level wrappers** (DistributedDataParallel, DistributedSampler) — simplified APIs for data-parallel training

Key insight: Distributed training in PyTorch is **data parallel by default**. Each process owns a full copy of the model; gradients are synchronized across processes via allreduce to enable jointly-trained models despite using partitioned data.

## Responsibilities

### What This Subsystem Owns

1. **Process Group Management** (`distributed_c10d.py`)
   - `init_process_group()`: establish communication between processes
   - `get_rank()`, `get_world_size()`: query process identity
   - Rank-to-device mapping: coordinate process-to-GPU assignment
   - Backend abstraction: support NCCL (GPU), Gloo (CPU/GPU), MPI (HPC)

2. **Collective Operations** (`distributed_c10d.py`, `_functional_collectives.py`)
   - `all_reduce()`: sum/average tensors across processes
   - `broadcast()`: send from one process to all
   - `scatter()`, `gather()`: distribute/collect tensor partitions
   - `barrier()`: synchronization point
   - Ring/tree topologies: optimize communication patterns

3. **Process Group Variants** (`distributed_c10d.py`)
   - Default group: all processes
   - Subgroups: create logical groups for partial synchronization
   - Grouped allreduce: synchronize only specific process subsets

4. **Distributed Data Loading** (`sampler.py`)
   - `DistributedSampler`: partition dataset non-overlappingly across ranks
   - Epoch-based shuffling: ensure different shuffles each epoch
   - Drop-last option: handle datasets not divisible by world_size

5. **High-Level Wrappers** (`parallel/distributed.py`)
   - `DistributedDataParallel (DDP)`: wrap modules, register backward hooks for gradient synchronization
   - Automatic gradient averaging: allreduce + divide by world_size
   - Bucketing: group gradients to amortize communication overhead
   - Pipelined synchronization: overlap communication with earlier-layer computation

6. **Communication Backend Initialization** (C++ layer, `_C` bindings)
   - NCCL backend: GPU-to-GPU communication via NVIDIA collectives
   - Gloo backend: CPU and GPU via Gloo library
   - MPI backend: high-performance interconnects
   - Device discovery: detect GPU topology for efficient routing

### What This Subsystem Does NOT Own

- **Model Definition**: torch.nn owns model layers and parameter organization
- **Gradient Computation**: torch.autograd computes gradients; distributed only synchronizes
- **Optimizer Updates**: torch.optim applies updates; distributed ensures gradients are synchronized first
- **Data Loading**: torch.utils.data owns dataset and dataloader; distributed only provides sampler
- **Device Memory Management**: torch.cuda owns device memory; distributed uses it for communication buffers
- **Single-Process Training**: if world_size is 1, distributed is no-op

## Dependencies

### Upstream Dependencies (What Uses This)

- **User Training Code**: DDP-wrapped models, allreduce calls in custom training loops
- **torch.nn.parallel.DistributedDataParallel**: wraps modules and registers gradient sync hooks
- **Large-scale training frameworks**: Hugging Face Transformers, Lightning, etc.
- **Checkpointing**: User code calls `get_rank()` to avoid duplicate checkpoints from all processes

### Downstream Dependencies (What This Uses)

- **torch.autograd**: Receives backward hooks that trigger allreduce; doesn't modify autograd
- **torch.nn.Module**: Wraps modules; reads parameters and registers backward hooks
- **torch.utils.data**: Uses DataLoader; provides DistributedSampler for partitioning
- **torch._C**: C++ bindings for NCCL, Gloo, MPI backends
- **System networking**: Underlying communication via TCP/IP, NVLink, Infiniband

### Dependency Direction

```
User Training Code
    ↓
torch.nn.parallel.DistributedDataParallel
    ↓
torch.distributed.init_process_group()
torch.distributed.all_reduce() [during backward hooks]
    ↓
NCCL/Gloo/MPI backends (C++)
    ↓
Network (TCP, NVLink, Infiniband)
```

## Trade-offs and Design Decisions

### Data Parallelism vs. Model Parallelism

**Decision**: PyTorch's distributed module prioritizes **data parallelism** (each process owns full model, partitions data).

**Trade-off**:
- ✅ **Advantage**: Simple API: wrap model with DDP, use DistributedSampler, training loop identical
- ✅ **Advantage**: Efficient for ~8-16 GPUs per machine
- ❌ **Disadvantage**: Each process uses full model memory; inefficient for models larger than single GPU
- ❌ **Disadvantage**: Model parallelism (pipeline or tensor parallelism) requires custom code

**Evidence**: Chapter 08 demonstrates DDP as the recommended approach; model parallelism not in core distributed package.

### Synchronous vs. Asynchronous Gradient Updates

**Decision**: DDP uses **synchronous updates** — all processes wait for allreduce to complete before stepping.

**Trade-off**:
- ✅ **Advantage**: Guarantees convergence properties of single-GPU training hold
- ✅ **Advantage**: Reproducible across runs; easier to debug
- ❌ **Disadvantage**: Slowest process becomes bottleneck (synchronization barrier)
- ❌ **Disadvantage**: No benefit from faster processes; strict lockstep

**Evidence**: `DistributedDataParallel._reducer` (parallel/distributed.py) implements barrier synchronization before step.

### Per-GPU Gradient Accumulation

**Decision**: Each process accumulates gradients from its local batch, then allreduce averages.

**Trade-off**:
- ✅ **Advantage**: No duplicate computation; each process sees only its data
- ✅ **Advantage**: Gradient averaging is cheap (single allreduce)
- ❌ **Disadvantage**: Effective batch size is world_size * local_batch_size (users must increase learning rate or adjust schedules)
- ❌ **Disadvantage**: Different random seed per process produces slightly different gradients (before averaging)

**Evidence**: Example in Chapter 08 shows each rank processes different data; gradients averaged post-backward.

### Allreduce Topology

**Decision**: Use tree or ring topology for allreduce rather than star (one process receives all).

**Trade-off**:
- ✅ **Advantage**: Communication load distributed; no single bottleneck
- ✅ **Advantage**: Bandwidth scales near-linearly with process count
- ❌ **Disadvantage**: More complex implementation
- ❌ **Disadvantage**: Ring topology has latency proportional to world_size

**Evidence**: NCCL backend implements ring and tree algorithms; Gloo supports multiple topologies.

### Explicit Initialization

**Decision**: User must explicitly call `init_process_group()` before using distributed operations.

**Trade-off**:
- ✅ **Advantage**: Clear that distributed initialization is happening
- ✅ **Advantage**: Flexible: can disable distributed for single-GPU testing
- ❌ **Disadvantage**: Boilerplate; easy to forget in multi-file projects
- ❌ **Disadvantage**: Error messages unclear if not called

**Evidence**: Chapter 08 training script shows explicit `dist.init_process_group()` call in main().

## Extension Boundaries

### Custom Collective Operations

**Boundary**: Register new collectives by implementing the backend interface (NCCL/Gloo/MPI).

Current collectives: allreduce, broadcast, scatter, gather, reduce, allgather, barrier, reduce_scatter, send, recv.

Adding new collectives requires C++ implementation and binding; not exposed to Python directly.

**Evidence**: `_functional_collectives.py` wraps C++ collective implementations.

### Using Subgroups

**Boundary**: Create process subgroups for hierarchical or partial synchronization.

```python
# Rank 0-1 in group A, rank 2-3 in group B
ranks_group_a = [0, 1]
group_a = dist.new_group(ranks=ranks_group_a)

# Only group A processes participate
dist.all_reduce(tensor, group=group_a)
```

**Evidence**: `new_group()` API (distributed_c10d.py); enables selective synchronization.

### Custom Sampler Integration

**Boundary**: Use `DistributedSampler` or implement custom sampler respecting rank/world_size.

```python
class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # Partition dataset
    
    def __iter__(self):
        # Yield only samples for this rank
        indices = dataset_indices[self.rank::self.world_size]
        return iter(indices)
```

**Evidence**: `DistributedSampler` in `sampler.py` provides reference implementation.

## Runtime Implications

### Lifecycle and Initialization

1. **Pre-training**: Launch multiple processes (e.g., `torch.distributed.launch`, Kubernetes)
2. **Initialization**: Each process calls `init_process_group()`, discovers peers, exchanges metadata
3. **Barrier**: All processes synchronize; any process can proceed only after all initialized
4. **Training**: Identical code on all processes (different data via DistributedSampler)
5. **Cleanup**: `destroy_process_group()` at end

### Concurrency Behavior

**Thread Safety Within Process**:
- **Unsafe**: Multiple threads within a process calling distributed operations simultaneously
- **Guidance**: Single training thread per process; use locks if necessary

**Cross-Process Synchronization**:
- **Synchronous**: Collectives block until all processes participate
- **Barrier**: If one process hangs, others block indefinitely (use timeout)
- **Evidence**: `all_reduce()` is blocking; no non-blocking mode in core API

### Failure Behavior

1. **Process Crash**: Other processes hang at barrier or collectives
2. **Network Partition**: Collective operations timeout or deadlock
3. **Rank Mismatch**: If processes disagree on world_size, initialization fails
4. **Backend Unavailable**: NCCL failure (e.g., GPU unavailable) raises error

**Evidence**: `init_process_group()` has timeout parameter (default 30 minutes); set via environment or parameter.

## Performance Implications

### Known Hotspots

1. **Allreduce Communication**: Largest overhead; scales with gradient tensor size
2. **Synchronization Barrier**: Fastest process waits for slowest
3. **Per-Process Overhead**: Each process executes full training loop; no speedup if not data-parallel

### Allocation Patterns

- **Communication Buffers**: Allocated by NCCL/Gloo backends; typically ring buffers for efficiency
- **Gradient Bucketing**: DDP groups gradients to reduce allreduce calls (trades memory for communication efficiency)

### Bandwidth and Latency

- **Allreduce Latency**: Proportional to log(world_size) for tree, world_size for ring
- **Bandwidth Utilization**: Depends on backend and network; NCCL maximizes GPU interconnect bandwidth
- **Effective Training Throughput**: actual_throughput = (data_throughput * world_size - communication_overhead) / world_size

## Ownership Boundaries

### State Owned by torch.distributed

1. **Process Group Registry**: Mapping of group names to processes
2. **Rank and World Size**: Process identity and group size
3. **Backend State**: NCCL context, Gloo store, MPI communicator
4. **Communication Buffers**: Temporary tensors for collective operations

### State Borrowed from Autograd/NN

1. **Gradients**: `.grad` attributes of parameters; owned by autograd
2. **Parameters**: Tensor data; owned by ATen allocator
3. **Model**: Module structure; owned by torch.nn

### State Owned by Users

1. **Process Launch**: Responsibility to launch correct number of processes
2. **Rank Assignment**: Environment variables (RANK, WORLD_SIZE)
3. **Synchronization Points**: User calls `barrier()` if additional synchronization needed

## Key Implementation Files

| File | Purpose |
|---|---|
| `distributed_c10d.py` | Core API: init_process_group, collectives, groups |
| `_functional_collectives.py` | Functional collective operations (tensors, scalars) |
| `_functional_collectives_impl.py` | Backend dispatch for collectives |
| `parallel/distributed.py` | DistributedDataParallel wrapper, gradient synchronization |
| `sampler.py` | DistributedSampler for data partitioning |
| `launcher.py` | Process launch helpers |
| `rendezvous.py` | Process discovery and rendezvous |
| `c10d/` (C++) | NCCL, Gloo, MPI backend implementations |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/distributed/` — checked for API, initialization, collective operations
2. **Book Chapter**: Chapter 08 "Distributed Training: Orchestration and Collectives" provides architectural understanding
3. **Code Flow**: Traced from user `init_process_group()` call through DDP gradient synchronization hooks to collective operations

Last Verified: 2026-05-27
