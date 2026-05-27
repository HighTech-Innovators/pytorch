# Architecture Decision Record: Distributed Training (torch.distributed)

## Architectural Role

**torch.distributed** is PyTorch's library for multi-process and multi-node training — the infrastructure for scaling model training across multiple GPUs or machines. It provides process group management, collective communication operations, and distributed data parallel wrappers for scaling training across resources.

**Location**: `torch/distributed/` | **Language**: Python frontend, C++ backend (NCCL, Gloo, MPI) | **Dependencies**: c10, ATen, torch

## Responsibilities

**torch.distributed owns**:
- Process group management (`init_process_group()`, `get_rank()`, `get_world_size()`)
- Collective communication operations (all-reduce, broadcast, gather, scatter, all-gather)
- Backend abstraction (NCCL for NVIDIA GPUs, Gloo for CPU/GPU cross-platform, MPI for HPC)
- Distributed data parallel wrapper (`nn.parallel.DistributedDataParallel`)
- Communication backends and protocol handling
- Process spawning and initialization (launcher)
- Gradient synchronization and communication

**torch.distributed does not own**:
- Actual GPU-to-GPU communication (NCCL, Gloo libraries own that)
- Model training loops (user code owns that)
- Parameter management (torch.nn owns that)
- Autograd (autograd owns that)

## Dependencies

### Inbound Dependencies
- **Multi-process training code** calls `dist.init_process_group()` to initialize
- **Training loops** call collective operations to synchronize gradients
- **Models** wrapped with `nn.parallel.DistributedDataParallel` for data parallelism
- **Data loaders** use `DistributedSampler` to partition data across processes

### Outbound Dependencies
- **NCCL** (external library) for GPU-to-GPU communication
- **Gloo** (external library) for cross-platform communication
- **MPI** (external library) for HPC cluster communication
- **torch.nn** for accessing model parameters
- **torch** for Tensor type

## Trade-offs and Design Decisions

### 1. Process Groups for Communication Abstraction
**Decision**: Processes organized into named groups; operations target specific groups.

**Rationale**: 
- Multiple independent groups can exist (e.g., one for data parallel, one for model parallel)
- Enables nested parallelism: data parallel within nodes, model parallel across nodes
- Default group "world" includes all processes

**Example**:
```python
dist.init_process_group(backend='nccl')  # Create "world" group with all processes
my_group = dist.new_group([0, 1, 2, 3])  # Create subgroup for subset of processes
dist.all_reduce(tensor, group=my_group)  # All-reduce within subgroup only
```

**Trade-off**: Complexity for users; single group sufficient for most use cases.

### 2. Backend Abstraction (NCCL, Gloo, MPI)
**Decision**: Pluggable backend interface; same API regardless of underlying communication library.

**Rationale**: 
- NCCL: optimized for NVIDIA GPU-to-GPU, fastest for multi-GPU
- Gloo: cross-platform CPU/GPU communication, good for heterogeneous clusters
- MPI: standard for HPC, integrates with cluster job schedulers
- User chooses backend via `init_process_group(backend='nccl'|'gloo'|'mpi')`

**Evidence**: torch/distributed/backend abstraction in C++ code.

**Trade-off**: Feature set differs by backend; some operations unavailable on some backends.

### 3. Synchronous All-Reduce for Gradient Averaging
**Decision**: All-reduce tensors across processes synchronously to average gradients.

**Rationale**: 
- Data parallelism requires synchronized gradient averaging
- All-reduce is standard collective operation in MPI/NCCL
- Synchronous ensures all processes proceed at same pace

**Alternative**: Asynchronous gradient updates (gossip, decentralized SGD) — harder to implement, less commonly used.

**Implementation**: Ring all-reduce algorithm (O(n) communication for n processes, not O(n²)).

### 4. DistributedDataParallel (DDP) Wrapper for Models
**Decision**: Provide `nn.parallel.DistributedDataParallel` to wrap models for data parallelism.

**Rationale**: 
- Automatically synchronizes gradients across processes after backward pass
- Overlaps gradient communication with backward pass computation
- Hides communication complexity from user

**Usage**:
```python
model = nn.Linear(10, 1)
ddp_model = nn.parallel.DistributedDataParallel(model)
```

**Trade-off**: Wrapping adds overhead; must plan data splitting across processes.

### 5. DistributedSampler for Data Splitting
**Decision**: Provide `DistributedSampler` to partition data across processes without overlap.

**Rationale**: 
- Each process must see disjoint data subset (no duplication)
- Sampler ensures deterministic sharding
- Enables proper statistical averaging across epochs

**Usage**:
```python
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### 6. Init Method: How Processes Find Each Other
**Decision**: Support multiple init methods for process discovery: environment variables, TCP master, file sharing.

**Rationale**: 
- `env://`: environment variables set by job scheduler (most common)
- `tcp://`: explicit master node and port (useful for debugging)
- `file://`: file-based synchronization (works without network)

**Example**:
```python
dist.init_process_group(backend='nccl', init_method='env://')
```

### 7. Blocking Communication Operations
**Decision**: All collective operations block until complete; optional asynchronous API via work objects.

**Rationale**: 
- Synchronous: easy to reason about ordering and completion
- Asynchronous: advanced users can overlap communication with computation

**Example**:
```python
# Blocking
dist.all_reduce(tensor)

# Non-blocking
work = dist.all_reduce(tensor, async_op=True)
work.wait()
```

### 8. Backend-Specific Communication
**Decision**: NCCL uses GPU-to-GPU communication; Gloo can use CPU or GPU depending on device type.

**Rationale**: 
- NCCL: specialized for NVIDIA GPU interconnect (NVLink, PCIe), extremely fast
- Gloo: general-purpose, slower but portable
- Choose backend based on hardware

**Trade-off**: NCCL faster but requires NVIDIA GPUs; Gloo works with AMD, Intel, CPU-only.

## Extension Boundaries

**Custom communication patterns**: Use low-level `recv()`, `send()`, `P2P` operations to implement custom patterns (e.g., ring reduce).

**Model parallelism**: Partition model layers across processes; more complex than data parallelism, not built-in to distributed.

**Pipeline parallelism**: Implement via custom communication patterns and gradient accumulation.

**Mixed precision distributed**: Use `torch.cuda.amp.GradScaler` with DDP to scale gradients during FP16 training.

## Runtime Implications

### Initialization
1. Job launcher (torch.distributed.launch or torchrun) spawns N processes
2. Each process calls `dist.init_process_group(backend='nccl', init_method='env://')`
3. Processes discover each other via init_method
4. Barrier synchronization: all processes must reach init before proceeding

**Time**: ~100ms-1s depending on number of processes and network.

### Synchronization: All-Reduce Algorithm
- Ring all-reduce: N-1 rounds, each process sends/receives (n elements / (N-1) processes)
- Time: O(N) with bandwidth inversely proportional to N

**Example**: 8 GPUs, 1GB tensor
- Each GPU sends/receives (1GB / 7) ≈ 140MB
- Time ≈ 140ms (assuming 1GB/s bandwidth)

### Gradient Synchronization in DDP
1. Forward pass on each GPU
2. Backward pass computes gradients locally
3. During backward: gradient all-reduce overlaps with remaining layers' backward passes
4. After backward: gradients synchronized, ready for optimizer step

**Overlap**: If overlapped perfectly, all-reduce time is hidden by backward computation.

### Memory
- All-reduce buffers: ~tensor size
- DDP state: bucketing, communication schedule
- No significant overhead beyond gradient tensors

### Concurrency
- **Synchronization points**: init, all_reduce, broadcast, etc.
- **All processes must reach** synchronization point or hang
- **Deadlock risk**: if any process fails to call all_reduce, entire job hangs

### Lifecycle
- Process group initialized once at start
- Processes run together until job completes
- Failure of one process kills entire job (no fault tolerance built-in)

## Performance Implications

### Known Hotspots
1. **All-reduce communication time**: Dominated by network bandwidth
2. **Gradient synchronization**: In data parallel, ~30-50% of time can be communication
3. **Process startup**: Spawning many processes adds overhead

### Optimization Opportunities
1. **Gradient accumulation**: Update less frequently, reduce all-reduce calls
2. **Communication overlap**: Overlap backward computation with all-reduce
3. **Local gradient accumulation**: Accumulate gradients locally before all-reduce
4. **Mixed precision**: Use FP16 for gradients, reduce communication bandwidth
5. **Gradient compression**: Quantize gradients before all-reduce (experimental)

### Scalability
- Strong scaling: time ∝ 1/(N * efficiency)
- Efficiency decreases with more processes (communication overhead dominates)
- Typical efficiency: ~90% for 8 processes, ~70% for 64 processes, ~30% for 1000 processes

### Network Dependency
- Single-node (8 GPUs on same machine): minimal communication overhead
- Multi-node: network bandwidth is bottleneck
- Slow networks: all-reduce can take 100-1000ms per step

## Ownership Boundaries

**torch.distributed owns**:
- Process group management and discovery
- Collective communication abstraction and implementation
- DDP wrapper and gradient synchronization
- Data sampling and distribution

**torch.distributed delegates to**:
- NCCL/Gloo/MPI backends for actual GPU-to-GPU/network communication
- torch.nn (for module wrapping)
- autograd (for gradient computation)

**User code owns**:
- Training loop and data loading
- Model definition and optimizer setup
- Process launching and job coordination

**Related systems own**:
- torch.distributed.launch (process launcher)
- torchrun (newer unified launcher)
- SLURM/PBS (cluster job schedulers)
