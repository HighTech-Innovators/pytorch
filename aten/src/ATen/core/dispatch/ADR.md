# Architecture Decision Record: aten/src/ATen/core/dispatch

## Architectural Role

`aten/src/ATen/core/dispatch` implements the core dispatcher routing mechanism that determines which kernel executes for each operator call. It manages dispatch key priority, fallback resolution, and the runtime dispatch table. This module is Runtime Critical because every single tensor operation (billions per training run) traverses this dispatch mechanism. The Dispatcher singleton defined here is the central routing nexus of the entire PyTorch execution engine.

## Responsibilities

This module is responsible for:
- Implementing the `Dispatcher` singleton that manages all operator kernels globally
- Managing dispatch key precedence ordering (Autograd > ADInplaceOrView > Backend-specific)
- Implementing the dispatch lookup algorithm: for a given operator and DispatchKeySet, determine which kernel to invoke
- Managing backend fallback kernels (catch-all kernels that apply to all operators for a given dispatch key)
- Implementing the LeftRight lock-free concurrent table for operator lookup
- Handling cache management and invalidation when new kernels are registered
- Supporting dynamic kernel registration and deregistration (though most registration is static)

The module does **not** implement concrete kernels (those live in aten/src/ATen/native) or define operator schemas (that's aten/src/ATen/core).

## Dependencies

**Inbound** (what depends on aten/src/ATen/core/dispatch):
- Every operator call in PyTorch: the dispatcher is invoked for every `add()`, `matmul()`, etc.
- `aten/src/ATen/native` for kernel implementations that are registered against dispatch keys
- `torch/csrc/autograd` for autograd fallback registration
- Python bindings for operator resolution

**Outbound** (what aten/src/ATen/core/dispatch depends on):
- `c10/core` for DispatchKeySet and DispatchKey enumerations
- `c10/util` for intrusive pointers and exception handling
- Standard library (STL for hash tables, atomics)

## Trade-offs

**Singleton pattern for global dispatcher**: There is one global Dispatcher instance, inlined as a static getter for zero overhead in steady-state. This eliminates parameter passing but couples all operators to a global object. The alternative (per-namespace dispatchers) would reduce contention but complicate lookup.

**Priority-based dispatch key ordering**: When multiple functionality keys are active (e.g., Autograd + CPU), the dispatcher selects the highest-priority one (Autograd wins). This ensures consistent layering but requires careful priority management. The alternative (multiple parallel dispatch paths) would be more flexible but more complex.

**Fallback kernels for missing entries**: If no kernel is registered for a specific operator-key combination, the dispatcher consults fallback kernels (e.g., Autograd fallback). This allows shared behavior across many operators without per-operator implementation but adds a lookup cost for operations with no explicit implementation.

**LeftRight lock-free table for operator lookup**: The operator lookup table uses LeftRight concurrency, allowing readers to proceed lock-free while writers update the table. This provides high read throughput but writer throughput is limited to one at a time. This is acceptable because operator registration is rare after library initialization.

## Extension Boundaries

- **Custom backend registration**: New hardware backends can register themselves as dispatch keys and provide fallback kernels that apply to all operators, enabling rapid integration without per-operator implementation.
- **Middleware dispatch layers**: New dispatch keys can be inserted between Autograd and Backend layers, enabling features like automatic mixed precision or tensor interception without modifying existing kernels.
- **Custom dispatch strategies**: Advanced users can register custom dispatch functions for specific operators to override default routing.

## Runtime Implications

**Dispatch cost**: Each operator call performs O(1) lookup in the DispatchKeySet to find the next dispatch key, then O(1) lookup in the operator table to find the kernel. Total overhead is typically 2-4 CPU cycles per call, amortized over the kernel execution time.

**Kernel caching**: The dispatcher maintains a cache of recent kernels to avoid repeated lookups for the same operator-key combination. Cache hits are extremely common in training loops (same operations called thousands of times per batch).

**Fallback resolution**: If a kernel is not found for the highest-priority dispatch key, the dispatcher descends to the next-lower-priority key until a kernel or fallback is found. This can involve 3-5 lookups in the worst case, but is typically 1-2 in practice.

**Lock-free reads**: Multiple threads can query the dispatcher concurrently without blocking. Writers (kernel registration) acquire exclusive locks, but registration is rare after startup.

## Performance Implications

**Dispatch overhead**: The dispatch lookup is part of the critical path for every operation. PyTorch's inlining strategy ensures that simple operations' dispatch is competitive with bare C++ function calls.

**Kernel caching**: Caching eliminates redundant lookups in tight loops, reducing per-call overhead from a few cycles to near-zero after the first call.

**Priority ordering**: By placing Autograd at the highest priority, the dispatcher ensures that autograd recording happens before the actual computation, adding zero overhead for operations that don't require gradients (no Autograd dispatch entry is consulted).

**Backend locality**: Backend-specific kernels are clustered by dispatch key, improving CPU cache locality when accessing kernel pointers.

## Ownership Boundaries

- **Dispatcher owns**: global operator registry, per-operator kernel tables, fallback kernel registry
- **OperatorEntry owns**: per-dispatch-key kernel pointers for that operator
- **KernelFunction owns**: the actual kernel code (but is registered and owned by its source module)
- **DispatchKeySet (per tensor) owns**: the set of active keys for that tensor (used to select which dispatcher entry to consult)

## Verification Points

- `aten/src/ATen/core/dispatch/Dispatcher.h:70-130` — Dispatcher singleton definition and operator table
- `aten/src/ATen/core/dispatch/Dispatcher.h:237-298` — Kernel and fallback registration interfaces
- `c10/core/DispatchKey.h:35-50` — Dispatch key precedence ordering
- `aten/src/ATen/core/dispatch/OperatorEntry.h` — Per-operator kernel table management
