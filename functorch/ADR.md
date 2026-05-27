# Architecture Decision Record: Function Transforms (functorch)

## Architectural Role

**functorch** is a JAX-like library providing composable function transforms for PyTorch. It enables efficient computation of per-sample gradients, batching (vmap), automatic differentiation variants (vjp, jvp), and higher-order derivatives. It builds on top of PyTorch's autograd and ATen to provide functional programming primitives that compose well with each other.

**Location**: `functorch/` | **Language**: Python + C++ binding | **Dependencies**: autograd, ATen, torch

## Responsibilities

**functorch owns**:
- `vmap` (vectorized map): batches operations across dimension
- `grad` and `grad_transform`: gradient computation
- `vjp` (vector-jacobian product): reverse-mode AD
- `jvp` (jacobian-vector product): forward-mode AD
- Composition of transforms (e.g., `vmap(grad(...))` for per-sample gradients)
- Batching rules for ATen operations
- FX-based tracing of transforms for compilation
- Performance optimization for composed transforms

**functorch does not own**:
- Core autograd (autograd engine owns that)
- Actual kernel implementations (ATen owns those)
- Tensor operations (ATen owns those)
- Module system (torch.nn owns that)

## Dependencies

### Inbound Dependencies
- **Research code** using per-sample gradients (functorch.grad)
- **Ensemble inference** using vmap to batch multiple models
- **Jacobian/Hessian computation** using composed transforms
- **JAX-style code** ported to PyTorch

### Outbound Dependencies
- **autograd** for gradient computation (vjp, jvp)
- **ATen** for tensor operations (batching rules must exist for each operation)
- **torch.nn** for module support (partial, with limitations)
- **FX** for tracing (torch.fx)

## Trade-offs and Design Decisions

### 1. Composable Transforms vs Single-Purpose API
**Decision**: Provide composable transforms (vmap, grad, vjp, jvp) rather than special-case functions (e.g., per_sample_grad as one function).

**Rationale**: 
- JAX model: composable transforms handle many use cases
- Same API for simple (single grad) and complex (vmap(grad(...))) cases
- Extensible: new transforms can be composed with existing ones

**Alternative**: Special-cased functions (simpler, but less flexible)

**Trade-off**: Learning curve steeper; composition requires understanding semantics of each transform.

### 2. Batching Rules for Each Operation
**Decision**: Each ATen operation requires explicit batching rule (how to handle vmap along batch dimension).

**Rationale**: 
- Operations have different semantics; generic batching won't work for all (e.g., reduce operations)
- Explicit rules enable optimization (fuse batch dimension into operations)
- Rules are relatively simple; most operations have direct batch rule

**Evidence**: functorch/_src contains batching rules for ATen operations.

**Trade-off**: Every new ATen operation needs batching rule; maintenance burden.

### 3. Lazy Batching Dimension Tracking
**Decision**: Track batch dimensions symbolically; don't actually expand tensors.

**Rationale**: 
- Expanding tensors would multiply memory usage by batch size
- Lazy tracking: operations work on original tensors with metadata about batch dims
- Storage efficiency: O(1) memory overhead per batch dimension

**Alternative**: Eager expansion (simple, but memory-inefficient)

### 4. FX Tracing for Optimization
**Decision**: Use FX to trace through vmap/grad and capture as graph, then optimize.

**Rationale**: 
- Graph enables fusion: multiple batched operations fused into single kernel
- Compilation: generate optimized C++ code for common patterns
- Enables deployment: export to ONNX or TorchScript

**Trade-off**: Tracing only works for traceable code; data-dependent control flow fails.

### 5. Separate Module from torch (Under Development)
**Decision**: functorch is separate module imported as `from functorch import vmap`, not part of torch.autograd.

**Rationale**: 
- Experimental/under development; keep separate from stable torch API
- Allows rapid iteration and API changes
- Eventually may be integrated into torch.autograd

**Trade-off**: Users must import separately; not default in torch.

### 6. JAX API Compatibility Goal
**Decision**: API designed to be compatible with JAX's transforms where possible.

**Rationale**: 
- Lower switching costs for users familiar with JAX
- Share knowledge and best practices from JAX community
- Enable easier code porting

**Trade-off**: Must maintain PyTorch semantics (e.g., requires_grad, in-place operations) which differ from JAX.

## Extension Boundaries

**New batching rules**: Add rule for new ATen operation; system automatically uses it when that operation encountered in vmap.

**Custom transforms**: Implement custom transform by defining how it transforms other transforms (e.g., custom AD variant).

**Tracing-based compilation**: Use FX tracing to capture transformed graph, then compile to C++ or other target.

## Runtime Implications

### Initialization
- functorch loads transform implementations
- Batching rules registered for all known ATen operations
- FX tracer initialized

### Forward Pass with vmap
1. User calls `batched_fn = vmap(fn, in_dims=...)`
2. `batched_fn` is wrapper that intercepts tensor inputs
3. For each tensor, mark batch dimension metadata
4. Call `fn` with tensors marked
5. Operations check metadata; apply batching rule if dimension marked
6. Return output with batch dimension removed

**Overhead**: Metadata tracking, per-operation batching rule lookup (~5-10%).

### Forward Pass with grad
1. User calls `gradients, output = grad(fn)(x)`
2. Execute forward pass (computes autograd graph)
3. Execute backward pass (autograd)
4. Return gradients

**Overhead**: Same as autograd (normal backward pass).

### Composition: vmap(grad(...))
1. Trace through grad transform with vmap-marked tensors
2. For each autograd operation, apply vmap batching rule
3. Backward pass applies batching rules to grad operations

**Overhead**: Nested: batching overhead × grad overhead.

### Memory
- vmap: minimal overhead (metadata only, no expansion)
- grad with vmap: memory for gradient accumulation × batch size
- Checkpointing: can reduce memory for large batch sizes

### Concurrency
- **Not thread-safe** during tracing/execution of composed transforms
- **Safe** for reading completed tensors
- Individual vmap/grad calls sequential

### Lifecycle
- Transform created once
- Reused for multiple forward/backward passes
- State preserved across calls

## Performance Implications

### Known Hotspots
1. **Batching rule dispatch**: Lookup and apply rule for each operation
2. **Gradient computation**: vmap(grad) compounds overhead
3. **Composition overhead**: Nested transforms add overhead multiplicatively

### Optimization Opportunities
1. **FX compilation**: Trace composed transform, fuse operations, compile to C++
2. **Batching fusion**: Fuse multiple batched operations into single kernel
3. **Checkpointing**: Reduce memory for large batch sizes in vmap
4. **Partial evaluation**: Specialize transforms for specific input shapes at compile time

### Deployment Performance
- Tracing + compilation can achieve 2-10x speedup over naive nested implementation
- Per-sample gradients: vmap(grad) can be much faster than explicit loop

## Ownership Boundaries

**functorch owns**:
- Transform implementations (vmap, grad, vjp, jvp)
- Batching rules and infrastructure
- Composition semantics and tracing

**functorch references**:
- autograd (uses gradient computation)
- ATen (uses operations, relies on batching rules)
- torch (uses types)

**Parent/peer systems own**:
- torch.nn (torch.nn modules have limited functorch support)
- torch.fx (FX tracing used by functorch)
- torch.jit (integration for deployment)

**Future (planned integration)**:
- May be merged into torch.autograd as standard API
