# Architecture Decision Record: torch/csrc/autograd

## Architectural Role

`torch/csrc/autograd` implements the C++ autograd engine, the core of PyTorch's differentiation system. It manages the computation graph (Nodes and Edges), orchestrates the backward pass execution via the Engine, and handles gradient accumulation. This module is Runtime Critical; the performance and correctness of all gradient-based training depends on this engine.

## Responsibilities

- Implementing `Node` (graph vertices representing operations)
- Implementing `Edge` (graph connections between nodes)
- Implementing `Engine` (backward pass execution via thread pool)
- Managing `SavedVariable` for tensor preservation across forward/backward
- Implementing `GraphTask` for backward execution state
- Providing backward hooks for gradient transformation
- Supporting higher-order derivatives and graph preservation

## Dependencies

**Inbound** (what depends on torch/csrc/autograd):
- Python layer (`torch/autograd`)
- All operator implementations (for autograd registration)
- Distributed training (for gradient synchronization)

**Outbound** (what torch/csrc/autograd depends on):
- `c10/core` for TensorImpl and DispatchKeySet
- `aten/src/ATen/core/dispatch` for dispatcher
- Standard library (threading, memory management)

## Trade-offs

**Priority-based scheduling (sequence_nr)**: Backward nodes are executed based on sequence number (creation order), approximating reverse topological order. This is simple and cache-efficient but not optimal for all graph shapes. The alternative (true topological sort) would be more complex and require metadata tracking.

**Thread pool for backward execution**: The Engine uses a fixed thread pool to execute backward operations. This amortizes thread creation overhead but may be suboptimal for very wide or very deep graphs.

**SavedVariable reference counting**: Tensors saved for backward are wrapped in SavedVariable, which manages reference counting and version checking. This prevents use-after-free but adds small overhead.

## Extension Boundaries

- **Custom backward functions**: Users define custom Node subclasses via `autograd.Function` to implement custom gradients.
- **Gradient hooks**: Users can register hooks on Nodes to intercept/transform gradients.

## Runtime Implications

**Backward pass execution**: `loss.backward()` calls `Engine::execute()`, which creates a `GraphTask` and schedules NodeTasks on worker threads. The task completes when all nodes have been processed.

**Gradient accumulation**: Gradients from multiple edges to the same node are summed before the node executes.

**Graph preservation**: By default, the graph is freed after backward. `retain_graph=True` preserves it for additional backward passes (enabling higher-order derivatives).

**Thread coordination**: The Engine manages a thread pool that executes backward operations concurrently, synchronizing via atomic counters.

## Performance Implications

**Backward overhead**: The autograd engine adds 10-20% overhead compared to forward-only execution (typical for ResNet50 on ImageNet).

**Thread pool efficiency**: The thread pool is most efficient for wide graphs (many independent backward nodes). For sequential graphs, single-threaded execution may be faster, but the overhead is small.

**Memory overhead**: The computation graph is stored in memory; for models with many intermediate tensors, this can double memory usage during training.

**Gradient accumulation**: When a variable is used multiple times, gradients are summed. This is O(n) in the number of uses but is typically fast (few uses per variable).

## Ownership Boundaries

- **Node owns**: backward function implementation and saved tensors
- **Engine owns**: execution scheduling and gradient accumulation
- **GraphTask owns**: backward execution state (outstanding count, errors)
- **SavedVariable owns**: reference to saved tensor with version tracking

## Verification Points

- `torch/csrc/autograd/node.h:64-100` — Node definition and apply method
- `torch/csrc/autograd/engine.h:56-73` — NodeTask and execution infrastructure
- `torch/csrc/autograd/edge.h` — Edge definition
- `torch/csrc/autograd/engine.h:87-113` — Priority scheduling logic
