# `c10/mobile`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/mobile` provides CPU memory allocators optimised for mobile and embedded inference deployments, where the system allocator's aggressive memory reclamation behaviour causes page faults that harm latency.

## Key Files

| File | Purpose |
|---|---|
| `CPUCachingAllocator.h` / `CPUCachingAllocator.cpp` | Block-caching CPU allocator that retains freed buffers in a size-keyed map rather than returning them to the OS; scoped via `WithCPUCachingAllocatorGuard` |
| `CPUProfilingAllocator.h` / `CPUProfilingAllocator.cpp` | Two-phase allocator: a recording pass captures the allocation sequence and lifetimes into an `AllocationPlan`; a replay pass serves the same sequence from a single pre-allocated blob |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::CPUCachingAllocator` | Caching allocator; `allocate(size_t)` returns cached or freshly allocated blocks; `free_cached()` releases all retained blocks |
| `c10::WithCPUCachingAllocatorGuard` | RAII guard that installs the caching allocator for the current thread's scope |
| `c10::AllocationPlan` | Recorded allocation sequence: sizes, lifetimes, offsets into a single blob |
| `c10::AllocationPlanner` | Records allocations during a profiling pass; validates on replay |
| `c10::CPUProfilingAllocator` | Replay allocator that serves all allocations from a single pre-computed blob |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Implements the `c10::Allocator` interface; `DataPtr` return type |
| [c10/util](c10/util/ADR.md) | depends-on | `flat_hash_map`, `SmallVector`, export macros |
| Mobile inference runtime (external) | depended-on-by | Mobile runtimes install `CPUCachingAllocator` before running inference |

## Runtime Behaviour

`CPUCachingAllocator` maintains a `flat_hash_map` from allocation size to a list of free pointers, protected by a `std::mutex`. On `allocate`, it checks the map for a cached pointer of the exact size before falling back to `malloc`. On `free`, it inserts the pointer into the map rather than calling `::free`. `CPUProfilingAllocator` operates in two phases: during recording, `AllocationPlanner` intercepts every `allocate`/`free` call and records the size and lifetime (as a pair of allocation IDs) into `AllocationPlan`; during replay, all allocations are served from a single `malloc`-ed blob at pre-computed offsets, making the replay path allocation-free.

## Performance Profile

- **Allocation sites**: `CPUCachingAllocator` reduces OS-level allocation calls to at most one per unique size. The per-call `mutex` lock is the dominant cost on hot paths where multiple sizes are allocated in sequence.
- **Synchronization costs**: A single `std::mutex` guards the entire size-keyed free-list map in `CPUCachingAllocator`. This is a single-threaded-use component by design (the header documents it as mobile-only with scoped installation).
- **Data movement**: `CPUProfilingAllocator`'s replay mode allocates all tensors from one contiguous blob, improving cache locality for models with stable allocation patterns.
- **Redundant or repeated work**: The profiling allocator's validation mode re-runs the planner on each forward pass to detect deviations from the recorded plan; this doubles allocation overhead and is intended only for debugging.

## Design Rationale

The dual-allocator design separates the common case (retaining freed blocks to avoid OS reclaim) from the optimized case (offline planning for statically-shaped inference). The profiling allocator's two-phase approach requires that allocation patterns be deterministic across invocations — a constraint that holds for typical mobile inference but not for training. Both allocators are installed via thread-local guards rather than globally replacing the default allocator, allowing them to coexist with training-side allocators in the same process.
