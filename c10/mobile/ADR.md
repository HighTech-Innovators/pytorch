# `c10/mobile`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/mobile` provides mobile-specific CPU allocation strategies for inference workloads. It contains a scoped CPU caching allocator and a profiling allocator that records allocation lifetimes, formulates a reusable memory plan, and serves later allocations from one blob. Book Chapter 02 explains the general allocator contract; this directory specializes that contract for mobile CPU memory pressure and page-fault behaviour.

## Key Files

| File | Purpose |
|---|---|
| `CPUCachingAllocator.h` | Mobile-only caching allocator API, allocation maps, and scoped guard declaration |
| `CPUCachingAllocator.cpp` | Size-keyed allocation cache, thread-local guard state, and fallback freeing on allocation failure |
| `CPUProfilingAllocator.h` | `AllocationPlan`, `AllocationPlanner`, `CPUProfilingAllocator`, and profiling/validation/serving guards |
| `CPUProfilingAllocator.cpp` | Allocation lifetime recording, greedy offset planning, validation, and blob-backed allocation implementation |
| `build.bzl` | Build rule metadata for the mobile allocator sources |

## Public Interface

`CPUCachingAllocator` exposes `allocate`, `free`, `record_free`, and a virtual destructor, and `GetDefaultCPUCachingAllocator`, `GetThreadLocalCachingAllocator`, and `ThreadLocalCachingAllocatorEnabled` expose allocator state. `WithCPUCachingAllocatorGuard` installs a caching allocator for a scoped inference region. `AllocationPlan` records sizes, lifetimes, offsets, and total blob size. `WithProfileAllocationsGuard`, `WithValidateAllocationPlanGuard`, and `WithProfilingAllocatorGuard` switch the thread-local planner or profiling allocator into recording, validation, or planned-allocation mode.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | CPU allocation and free primitives from `c10/core/impl/alloc_cpu.h` |
| [c10/util](c10/util/ADR.md) | depends-on | `SmallVector`, `flat_hash_map`, `Exception`, and `irange` |
| Mobile inference runtime | depended-on-by | Scoped guards wrap model execution to reduce page faults or replay planned allocation layouts |

## Runtime Behaviour

`CPUCachingAllocator::allocate` locks a global mutex, looks up an exact-size pointer in `available_map_`, and returns cached memory when one exists; otherwise it calls `c10::alloc_cpu`, records the pointer in `allocation_map_`, and retries after `free_cached()` if allocation throws a c10 error. `CPUCachingAllocator::free` does not return allocator-owned memory to the OS immediately; it pushes the pointer into a size bucket so later allocations of the same size reuse it. `AllocationPlanner` records every allocation id and sets each lifetime to the id of the first later allocation after its free. `CPUProfilingAllocator` uses a validated plan to allocate from `blob_` at precomputed offsets instead of issuing a separate CPU allocation for each tensor buffer.

## Performance Profile

The caching allocator targets mobile platforms that aggressively return memory to the system and pay page faults on repeated inference allocations. Its exact-size buckets make reuse O(1) on a hash-table lookup plus vector pop, while the global mutex serializes public access to shared maps. The profiling allocator is explicitly not thread-safe, but it removes repeated allocation calls after profiling by serving all managed allocations from one contiguous blob. The greedy planner merges adjacent free blocks and assigns offsets only for allocations that are freed inside the profiled scope, so unmanaged lifetimes stay outside the plan.

## Design Rationale

Mobile inference has stable allocation patterns and tighter memory behaviour than server training, so this directory adds opt-in allocators instead of changing the general CPU allocator for every build. The caching allocator keeps the simple `allocate`/`free` shape and can wrap inference with one guard, which limits integration cost. The profiling allocator adds a three-step flow - profile, validate, execute - because a static memory plan is safe only when representative inputs produce the same allocation sequence. The implementation stays in c10 so mobile builds can use it below ATen and TorchScript code without Python dependencies.

