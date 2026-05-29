# Architecture Decision Record: c10/cuda

## Architectural Role

`c10/cuda` provides the CUDA-specific device primitives and memory management infrastructure for PyTorch. It implements the CUDACachingAllocator, which eliminates expensive GPU memory allocations during training via intelligent block pooling and reuse. This module bridges PyTorch's tensor abstraction with NVIDIA's CUDA runtime, managing device streams, events, and memory lifecycle. It is Runtime Critical for GPU-accelerated training and inference.

## Responsibilities

This module is responsible for:
- Managing CUDA device state and stream selection (primary stream, default stream per device)
- Implementing `CUDACachingAllocator` (5390 lines) which pools GPU memory blocks, performs best-fit allocation, and minimizes expensive cudaMalloc calls
- Enforcing per-stream block isolation to prevent data races without implicit synchronization
- Tracking memory statistics (allocation counts, peak usage, fragmentation) via `DeviceStats`
- Providing memory snapshot infrastructure for debugging (`torch.cuda.memory._snapshot()`)
- Managing CUDA events for stream synchronization and timing
- Implementing expandable segment support via CUDA virtual memory APIs (when configured)
- Handling out-of-memory scenarios with configurable observer hooks for custom error handling

The module does **not** implement compute kernels, operator dispatch (that's aten/src/ATen), or distributed collective communication (that's torch/distributed).

## Dependencies

**Inbound** (what depends on c10/cuda):
- `c10/core/Allocator.h` registration: c10/cuda provides the allocator implementation
- `torch/cuda` (Python layer) for memory management API exposure
- `aten/src/ATen/native/cuda` for kernel execution context
- `torch/distributed/nccl` for collective ops (uses streams allocated by c10/cuda)

**Outbound** (what c10/cuda depends on):
- NVIDIA CUDA runtime (cudaMalloc, cudaFree, cudaStream_t, cudaEvent_t)
- `c10/core` for Allocator interface and error handling
- System libraries for threading (mutex for block pool synchronization)

## Trade-offs

**Separate pools for large and small allocations**: Large allocations (>1MB) and small allocations (<1MB) use separate free-block pools. This prevents large allocations from fragmenting small-allocation pools but requires separate tracking logic. The alternative (unified pool) would be simpler but experience higher fragmentation in practice.

**Proactive over-allocation**: Requests between 1MB and 10MB trigger allocation of 20MB blocks (2x over-allocation). This reduces fragmentation and future allocation calls but consumes more peak GPU memory during training. The exact threshold is tuned via configuration.

**Per-stream block isolation without sync**: Blocks freed on stream A are not reusable on stream B without explicit synchronization. This prevents data races and is sound, but means peak GPU memory usage can be higher than if blocks were globally shared. The alternative (global pool with sync barriers) would reduce memory but increase latency.

**Best-fit allocation strategy**: The allocator searches for the smallest free block ≥ requested size, then splits it if oversized. This minimizes fragmentation but has O(n) search cost per allocation. The alternative (first-fit or power-of-two) would be O(1) but produce more fragmentation.

**Expandable segments (optional)**: CUDA virtual memory APIs allow segment growth without remapping when configured, reducing fragmentation at the cost of additional CUDA driver overhead for address space management.

## Extension Boundaries

- **Custom memory observers**: Callers can register `OutOfMemoryObserver` callbacks to monitor OOM events and trigger custom error handling (e.g., gradient checkpointing, model offloading).
- **Statistics export**: Memory statistics are exposed via `torch.cuda.memory_stats()` and can be accessed by profilers and monitoring systems.
- **Allocator configuration**: Threshold values (large/small boundary, split size limits, expandable segments flag) can be tuned at runtime.

## Runtime Implications

**Initialization**: When a CUDA tensor is first created on a device, the allocator is lazily initialized on that device. Subsequent allocations on the same device reuse the pool without reinitializing.

**Stream-aware lifecycle**: Each stream maintains its own pool of free blocks. When a block is freed on a stream, it is returned to that stream's pool. Blocks from different streams are never mixed without explicit event synchronization.

**OOM handling**: When cudaMalloc fails, the allocator attempts recovery:
1. Free the largest cached non-split block and retry
2. Free ALL cached non-split blocks and retry
3. Trigger Python garbage collection and retry
4. Raise OOM error

Custom observers are notified at each stage to enable graceful degradation.

**Statistics tracking**: Per-device statistics are accumulated (allocation counts, peak usage, fragmented bytes) and can be queried at runtime without stopping execution.

## Performance Implications

**Memory allocation amortization**: Packing small allocations into 2MB buffers amortizes the cost of cudaMalloc (hundreds of microseconds) across multiple tensor allocations, reducing latency variance in training loops.

**Fragmentation reduction**: Separate pools and proactive over-allocation reduce address space fragmentation, maintaining allocator performance even after billions of allocations in long training runs.

**Stream efficiency**: Per-stream pools enable independent memory management for different compute streams without contention, improving scaling for multi-GPU training.

**Search complexity**: Best-fit search is O(n) in the number of free blocks. In steady-state training, n is typically small (~10-100 blocks), making this acceptable. For workloads with many allocation sizes, this could become a bottleneck.

## Ownership Boundaries

- **CUDACachingAllocator owns**: stream-specific free-block pools, block metadata (size, address, stream association)
- **CUDACachingAllocator owns**: GPU memory segments (via cudaMalloc; raw pointers managed by UniqueVoidPtr)
- **StorageImpl owns**: the allocated block (via intrusive_ptr to allocator for deallocation)
- **Device statistics own**: per-device counters aggregated across all streams

## Verification Points

- `c10/cuda/CUDACachingAllocator.cpp:80-100` — Block pooling and reuse policies
- `c10/cuda/CUDACachingAllocator.h:74-110` — Statistics interface and memory snapshots
- `c10/cuda/CUDAStreamGuard.h` — Stream selection and context management
- `c10/cuda/CUDAException.h` — CUDA error handling and translation to PyTorch exceptions
