# Architecture Decision Record: PyTorch Autograd Engine

**Location:** `./src/torch/autograd/` (Python API) + `./src/torch/csrc/autograd/` (C++ implementation)  
**Last Updated:** 2026-05-27  
**Classification:** Runtime Critical, Coordination Heavy, State Owner

---

## Executive Summary

The Autograd engine is PyTorch's signature feature that enables automatic differentiation through computational graphs. It tracks which operations execute during forward pass, constructs a computational graph implicitly, and reverses that graph during backward pass to compute gradients. This is what distinguishes PyTorch from NumPy and enables deep learning.

---

## Architectural Role

Autograd serves as the **gradient computation orchestrator** for the entire framework. Its responsibilities are:

1. **Forward Graph Construction** — Track which operations execute and record them as nodes
2. **Backward Execution** — Reverse-topologically traverse the graph and compute gradients
3. **Gradient Accumulation** — Accumulate gradients from multiple paths through the graph
4. **Leaf Tensor Tracking** — Store gradients only in user-created (leaf) tensors
5. **Custom Operation Support** — Allow users to define custom forward/backward operations
6. **Gradient Mode Control** — Enable/disable gradient tracking via context managers

Because every gradient computation flows through autograd, **correctness and performance of training is fundamentally dependent on autograd's design**.

---

## Responsibilities

### What Autograd Does

- **Tape Recording** — During forward pass, each operation records itself (grad_fn) on the output tensor
- **Graph Construction** — Builds implicit DAG via grad_fn pointers; not materialized upfront
- **Backward Pass Execution** — Runs reverse topological sort starting from output tensor
- **Gradient Computation** — Calls each operation's backward() method to compute gradients
- **Accumulation** — Combines gradients from multiple paths (e.g., if a tensor is used twice)
- **Context Managers** — `torch.no_grad()`, `torch.enable_grad()` control gradient tracking
- **Leaf Tensor Gradients** — Stores computed gradients in `tensor.grad` attribute
- **Higher-Order Gradients** — Supports `grad_fn` on grad tensors for second derivatives
- **Debugging** — Provides anomaly detection and gradient checking utilities

### What Autograd Does NOT Do

- **Kernel Execution** — Kernels run via ATen; autograd only orchestrates backward
- **Memory Management** — Tensors and gradients managed by reference counting
- **Device Placement** — Device decisions made at tensor creation time
- **Operation Registration** — Operations registered in ATen; autograd only calls backward
- **Graph Optimization** — No optimization; graph is traversed as-is

---

## Dependencies

### What Depends On Autograd

- **torch/** — torch.tensor() sets requires_grad, .backward() calls autograd engine
- **torch/nn/** — nn.Module training loop calls .backward() and .step()
- **torch/optim/** — Optimizer reads tensor.grad and updates parameters
- **All User Code** — Training scripts call .backward() and access gradients

### What Autograd Depends On

- **ATen** — Calls aten:: kernels during forward pass
- **torch/csrc/autograd/** — C++ engine implementation
- **c10/core/** — Uses Tensor and Device types
- **PyTorch Extension API** — For custom backward implementations

**Dependency Direction**: Autograd sits in the middle. torch/ and user code depend on autograd; autograd depends on ATen for forward pass kernels.

---

## Trade-Offs and Design Decisions

### Decision 1: Implicit DAG via grad_fn Pointers vs Explicit Graph Materialization

**What**: Graph is not materialized upfront. Instead, each tensor carries a `grad_fn` pointer to the operation that created it; the graph is implicit in these pointers.

**Why**:
- Memory Efficiency: Don't allocate graph nodes until backward pass
- Dynamic Graphs: Graph structure can change on every forward pass
- Simple Reasoning: No "graph construction phase" — graph exists in grad_fn pointers
- Debugging: Gradcheck can re-run backward without materializing full graph

**Alternatives**:
- Explicit graph materialization (like TensorFlow 1.x static graphs) — requires explicit graph building phase
- Eager graph construction (store all nodes upfront) — wastes memory for graphs never used

**Trade-offs**:
- **Pro**: Memory efficient; supports dynamic graphs; simple implementation
- **Con**: Implicit graph harder to debug; no pre-pass optimization

---

### Decision 2: Reverse-Mode (Backward Mode) AD vs Forward-Mode AD

**What**: Gradients computed by reverse topological traversal from output to input (backward pass).

**Why**:
- Machine Learning Efficiency: Computing gradient of scalar loss w.r.t. 1M parameters is efficient in reverse-mode
- NumPy Compatibility: Matches NumPy's implicit scalar loss model
- Standard Practice: Backward-mode AD is the de-facto standard for deep learning

**Alternatives**:
- Forward-mode AD — compute gradient of each input w.r.t. output; inefficient for many parameters
- Mixed-mode AD — use forward-mode for some operations, backward-mode for others

**Trade-offs**:
- **Pro**: Efficient for ML workloads; standard expectation
- **Con**: Can't easily compute Jacobian-vector products (Jvp); forward-mode would be better

---

### Decision 3: Leaf-Only Gradient Storage

**What**: Gradients are only stored in leaf tensors (created by user, not as output of operation); intermediate tensor gradients are not saved.

**Why**:
- Memory Efficiency: Avoid storing gradients for millions of intermediate tensors
- Simplicity: Users only care about parameter gradients; intermediate gradients auto-deleted
- Performance: Fewer tensors to allocate and deallocate

**Alternatives**:
- Store Gradients for All Tensors — would use massive amounts of memory
- Store Gradients on Demand — would require graph traversal to determine which tensors need gradients

**Trade-offs**:
- **Pro**: Very memory efficient; suitable for large models
- **Con**: Can't inspect intermediate gradients; requires .retain_grad() for non-leaf tensors

---

### Decision 4: Python API + C++ Engine Implementation

**What**: Python API (torch/autograd/__init__.py) wraps C++ engine (torch/csrc/autograd/).

**Why**:
- Performance: Gradient computation is hot path; C++ is necessary for speed
- User Convenience: Python API is easy to use for 99% of cases
- Extensibility: Users can extend via custom Function classes (Python)

**Alternatives**:
- Pure Python — too slow for production
- Pure C++ — hard for users to extend or debug

**Trade-offs**:
- **Pro**: Good balance of performance and usability
- **Con**: Python-C++ boundary adds complexity; more code to maintain

---

### Decision 5: Thread-Safe Engine with Reentrant Backward

**What**: Multiple backward() calls can be in progress concurrently; engine uses locks to manage state.

**Why**:
- Higher-Order Gradients: Computing gradients of gradients requires backward to be reentrant
- Parallel Training: Distributed training may call backward on multiple threads
- Safety: Prevents race conditions during gradient accumulation

**Alternatives**:
- Single-Threaded Engine — would block on concurrent backward calls
- Lock-Free Design — would be complex and error-prone

**Trade-offs**:
- **Pro**: Supports higher-order gradients and parallelism
- **Con**: Lock contention on dense workloads; potential for deadlock (mitigated with MAX_DEPTH=60)

---

## Extension Boundaries

### Extending Autograd

**Supported Extensions:**
1. **Custom Operations** — Inherit from `torch.autograd.Function` and implement forward/backward
2. **Custom Backward** — Override backward for existing operations via hooks
3. **Gradient Hooks** — Register hooks on tensor gradients to intercept/modify before backprop
4. **Disable/Enable Gradients** — Use `torch.no_grad()`, `torch.enable_grad()` context managers

**NOT Supported:**
- Modifying graph structure after forward pass
- Changing backward computation order
- Custom AD modes beyond reverse-mode (forward-mode is limited)

### Integration Points

- **ATen** — Autograd calls ATen kernels during forward; ATen operations must be registered with autograd
- **torch/** — torch.Tensor class has .backward() and .grad attributes
- **torch/nn/** — nn.Module calls .backward() on loss
- **Custom Operations** — Users implement Function.backward() to define custom gradients

---

## Runtime Implications

### Initialization Order

**At Module Import:**
1. `torch/autograd/__init__.py` — Load Python autograd API
2. `torch/csrc/autograd/` — Initialize C++ engine, register default backward implementations
3. `torch/nn/` — Load nn.Module, which uses autograd

**At First Backward:**
- Graph traversal begins from output tensor
- Backward functions called in reverse topological order
- Gradients accumulated in leaf tensors

### Concurrency and Thread Safety

- **Graph Construction (Forward)** — Thread-safe; each thread builds its own subgraph via thread-local grad_fn
- **Backward Execution** — Serialized by lock; engine holds lock during entire backward pass
- **Gradient Accumulation** — Locked; prevents race conditions when accumulating gradients
- **Thread-Local Device Context** — Each thread can have its own "current device"

### Failure Modes

- **Leaf Tensor Not Holding Gradient** — Intermediate tensors don't save gradients; requires `.retain_grad()`
- **Backward Without Scalar Loss** — .backward() expects scalar output; non-scalar requires specifying gradient_output
- **Graph Deletion Due to Optimization** — Python garbage collection can delete grad_fn between operations
- **Deadlock Risk** — Reentrant backward with deeply nested operations can trigger MAX_DEPTH protection

---

## Performance Implications

### Hot Paths and Allocations

1. **Forward Graph Recording** — O(1) per operation; just store grad_fn pointer
2. **Backward Graph Traversal** — O(V+E) in DAG; V=nodes, E=edges
3. **Gradient Computation** — Depends on operation; matrix multiplication backward is expensive
4. **Gradient Accumulation** — O(1) per gradient; tensor addition

### Known Bottlenecks

- **Lock Contention** — Engine-wide lock on backward pass; not parallelizable within single backward()
- **Memory Allocation** — Creating intermediate gradient tensors during backward
- **Graph Traversal** — Large models with deep graphs have significant traversal overhead
- **Synchronization** — CUDA operations require device synchronization between forward and backward

### Mitigation Strategies

- **Gradient Checkpointing** — Trade compute for memory: recompute activations during backward instead of storing
- **In-Place Operations** — Minimize intermediate tensor allocations
- **Distributed Backward** — Split backward across multiple processes (not single-threaded)
- **Operation Fusion** — Combine operations to reduce backward traversal overhead

---

## Ownership Boundaries

### What Autograd Owns

- Graph recording during forward pass (grad_fn pointers)
- Backward pass execution and traversal
- Gradient accumulation logic
- Leaf tensor gradient storage (.grad attribute)
- Context managers for controlling gradient tracking

### What Autograd Does NOT Own

- Forward kernel execution (ATen's responsibility)
- Tensor memory management (reference counting in c10/core)
- Operation registration (ATen's responsibility)
- Python bindings (torch/ wrapper's responsibility)

### State Shared with Other Layers

- **grad_fn Pointers** — Stored in Tensor; visible to users
- **tensor.grad** — Stored in Tensor; readable by optimizers
- **Autograd Metadata** — Stored in TensorImpl via AutogradMetaInterface (c10/core)
- **Operation Records** — Each operation record includes backward function pointer

---

## Testing and Validation

### Critical Tests

- **Gradient Correctness** — torch.autograd.gradcheck compares numerical vs analytical gradients
- **Graph Construction** — Verify grad_fn pointers form correct DAG
- **Accumulation** — Verify gradients accumulate correctly from multiple paths
- **Leaf Gradient Storage** — Verify only leaf tensors store gradients
- **Higher-Order Gradients** — Verify gradients of gradients work correctly
- **Device Transfer** — Verify gradients transfer correctly between devices
- **Backward Without Scalar** — Verify errors when backward() called on non-scalar tensors

### Known Gaps

- No explicit concurrency tests for multi-threaded backward on contention
- Limited testing of very deep computation graphs (potential MAX_DEPTH issues)
- No performance regression tests for common backward patterns

---

## Related Systems

- **ATen** — Provides forward kernels; autograd orchestrates their execution and backward
- **torch/nn/** — Module training loops call backward() and access gradients
- **torch/optim/** — Optimizers read gradients from tensor.grad and update parameters
- **torch/csrc/autograd/** — C++ engine implementation
- **torch/autograd/function.py** — User-facing Function class for custom backward

---

## References

- `torch/autograd/__init__.py` — Python autograd API
- `torch/autograd/function.py` — User-facing Function class for custom operations
- `torch/autograd/grad_mode.py` — Context managers for gradient tracking control
- `torch/csrc/autograd/` — C++ engine implementation (see engine.h, engine.cpp)
- Book Chapter 2 (Autograd) — Comprehensive explanation of autograd system
- Book Chapter 1 (Initialization) — How autograd is initialized during import
