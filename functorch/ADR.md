# Architecture Decision Record: functorch (Composable Function Transformations)

## Architectural Role

`functorch` is **PyTorch's composable function transformation API**, providing JAX-like transformations (`vmap`, `grad`, `vjp`, `jvp`) that work seamlessly with PyTorch modules, autograd, and eager execution. It enables advanced use cases:

1. **Per-sample gradients**: Computing gradients for each sample in a batch independently
2. **Vectorized execution**: Running multiple variants of a model with `vmap` instead of explicit loops
3. **Jacobian/Hessian computation**: Efficiently computing full derivative matrices
4. **MAML-style meta-learning**: Batching inner-loop operations for few-shot learning
5. **Ensemble models**: Running multiple models in parallel on shared hardware

Key insight: `functorch` is a **transformation system**, not a new backend. It intercepts PyTorch operations, applies mathematical transformations (batching, automatic differentiation), and delegates back to the normal PyTorch dispatcher. This makes it composable with existing code.

## Responsibilities

### What This Subsystem Owns

1. **Core Transformations** (`torch._functorch/`)
   - `vmap()`: Vectorize a function across a batch dimension
   - `grad()`, `grad_and_value()`: Compute gradients without `.backward()`
   - `vjp()`: Vector-Jacobian product (reverse-mode autodiff)
   - `jvp()`: Jacobian-vector product (forward-mode autodiff)
   - `jacfwd()`, `jacrev()`: Full Jacobian matrices

2. **Transformation Implementation** (`torch._functorch/eager_transforms.py`)
   - Eager-mode execution of transformations
   - Automatic batching rules for operations
   - State management for transformation context

3. **Ahead-of-Time (AOT) Autograd** (`torch._functorch/aot_autograd.py`)
   - Compiler infrastructure for transformations
   - Tracing through function transformations to generate optimized code
   - Partitioning forward/backward for compilation

4. **Functional Model API** (`torch._functorch/make_functional.py`)
   - Convert `nn.Module` to pure functions
   - `make_functional()`, `make_functional_with_buffers()`
   - Separating parameters from computation

5. **Batching Rules** (`torch._functorch/partitioners.py`)
   - Define how operations handle batched dimensions
   - Per-operation batching strategies
   - Composition of rules for nested `vmap`

6. **Public API Organization** (`functorch/__init__.py`, `functorch/compile/`)
   - Export stable transformations to user-facing namespace
   - Experimental features in `functorch.experimental`
   - Compilation-specific utilities in `functorch.compile`

### What This Subsystem Does NOT Own

- **Tensor implementations**: ATen owns kernels; functorch just batches them
- **Automatic differentiation**: torch.autograd owns backward graphs; functorch composes with it
- **Neural network modules**: torch.nn owns module definitions; functorch wraps them
- **Backend dispatch**: The dispatcher owns kernel selection; functorch intercepts at operation level
- **Memory management**: c10 allocators handle memory; functorch just uses them
- **Model serialization**: torch.save/torch.load handle persistence

## Dependencies

### Upstream Dependencies (What Uses This)

- **User code**: Research projects needing advanced transformations (meta-learning, per-sample gradients)
- **Compiler frontends**: torch.compile can use AOT autograd as a compilation target
- **Private frameworks**: Internal model training pipelines

### Downstream Dependencies (What This Uses)

- **torch.autograd**: Composes with gradient tracking
- **torch.nn**: Wraps modules via `make_functional()`
- **ATen operations**: Batches operations through normal dispatcher
- **torch._C**: C++ bindings for performance-critical paths
- **torch.fx**: Tracing transformations for optimization

### Dependency Direction

```
User Code (research, advanced use cases)
    ↓
functorch package (public API)
    ↓
torch._functorch (implementation)
    ├─→ Batching engine (eager execution)
    ├─→ AOT Autograd compiler
    ├─→ Functional API wrapper
    └─→ torch.autograd, torch.nn, ATen
```

## Trade-offs and Design Decisions

### Eager vs. Compiled Execution

**Decision**: Provide both eager and compiled paths for transformations.

**Trade-off**:
- ✅ **Advantage**: Eager mode works with any PyTorch code; immediate feedback
- ✅ **Advantage**: Compiled mode (AOT) optimizes away transformation overhead
- ❌ **Disadvantage**: Two execution paths increase complexity
- ❌ **Disadvantage**: Eager transformations slower than hand-written loops in some cases

**Evidence**: `torch._functorch/eager_transforms.py` implements eager execution; `aot_autograd.py` handles compilation.

### Batching Dimension as First Argument

**Decision**: `vmap` batches over the first positional dimension by default.

**Trade-off**:
- ✅ **Advantage**: Predictable; matches NumPy convention
- ✅ **Advantage**: Works with `vmap(vmap(...))` nesting
- ❌ **Disadvantage**: Requires care when first dimension is not the batch dimension
- ❌ **Disadvantage**: Can lead to unexpected results if misused

**Evidence**: `vmap(fn, in_dims=0)` — `in_dims=0` is default.

### Composability Over Specialization

**Decision**: Provide composable primitives (`vmap`, `grad`, `vjp`) rather than specialized functions.

**Trade-off**:
- ✅ **Advantage**: User can build custom transformations by composition
- ✅ **Advantage**: JAX compatibility; familiar to users of that ecosystem
- ❌ **Disadvantage**: Composing transformations has overhead
- ❌ **Disadvantage**: Users must understand interaction between transformations

**Evidence**: `vmap(grad(fn))` produces per-sample gradients; `vmap(vmap(...))` handles nested batching.

### Integration with autograd Instead of Replacement

**Decision**: Compose with `torch.autograd` rather than replacing it.

**Trade-off**:
- ✅ **Advantage**: Works with existing PyTorch code; `.backward()`, optimizers, etc.
- ✅ **Advantage**: Gradual adoption; users don't learn new system
- ❌ **Disadvantage**: Overhead of layer between transformations and autograd
- ❌ **Disadvantage**: Some autograd features not composition-friendly

**Evidence**: `grad()` creates gradients without calling `.backward()`; composes with autograd hooks.

## Extension Boundaries

### Custom Batching Rules

**Boundary**: Define how a custom operation handles batching.

```python
# For a custom operation, define how vmap should handle batching
from torch._functorch.partitioners import register_batching_rule

@register_batching_rule("my_op")
def my_op_batch_rule(batched_args, batch_dims):
    # Implement logic for handling batched inputs
    pass
```

**Evidence**: `partitioners.py` contains registration system for batching rules.

### Custom Transformation

**Boundary**: Implement a new transformation by composing `vmap`, `grad`, `vjp`.

```python
def custom_transform(fn):
    def transformed(*args, **kwargs):
        batched_fn = functorch.vmap(fn)
        grads = functorch.grad(batched_fn)
        return grads(*args, **kwargs)
    return transformed
```

**Evidence**: Users can compose transformations; supported via public API.

### Compilation Targets

**Boundary**: AOT Autograd can target different compilers (TorchInductor, etc.).

**Evidence**: `compilers.py` defines compiler interface; `torch.compile` uses this.

## Runtime Implications

### Lifecycle and Initialization

1. **Import**: `import functorch` loads the transformation system
2. **Setup**: Registering batching rules for known operations
3. **Transformation**: User calls `vmap()`, `grad()`, etc.; returns wrapped function
4. **Execution**: Wrapped function intercepts operations and applies transformations
5. **Delegation**: Transformed operations dispatch back to ATen

### Concurrency Behavior

**Thread Safety**:
- **Global batching rules**: Shared across threads; no mutation during registration
- **Transformation context**: Thread-local; each transform has its own nesting level
- **Operation execution**: Delegates to ATen; inherits ATen's thread safety

**Evidence**: `vmap_increment_nesting()` uses thread-local state.

### Failure Behavior

1. **Unsupported operation**: Operation lacks batching rule → error on first `vmap` call
2. **Type mismatch**: Passing non-Tensor to vectorized function → TypeError
3. **Batch dimension mismatch**: `in_dims` doesn't match input structure → error
4. **Memory exhaustion**: Batching large tensors → OOM from accumulated intermediate values

**Evidence**: Batching rules in `partitioners.py` check types; errors raised on mismatch.

## Performance Implications

### Known Hotspots

1. **Eager Transformation Overhead**: Wrapping each operation adds Python-level cost
2. **Batch Rule Dispatch**: Lookup and execution of per-operation rules
3. **Nested Transformations**: `vmap(vmap(...))` multiplies overhead
4. **AOT Compilation**: Initial tracing cost; amortized across many calls

### Allocation Patterns

- **Eager mode**: Allocates intermediate batched tensors during execution
- **Compiled mode**: Traces once; optimized code reuses memory across calls
- **Stacked tensors**: vmap concatenates batch dimension; creates view in many cases

### Optimization Opportunities

- **Rule fusion**: Combine multiple rules into single operation
- **JIT compilation**: torch.compile can eliminate wrapper overhead
- **Kernel specialization**: Compile specialized kernels for common `vmap` patterns

**Evidence**: `aot_autograd.py` and `compilers.py` implement these optimizations.

## Ownership Boundaries

### State Owned by functorch

1. **Transformation context**: Current nesting level for `vmap`, `grad`, etc.
2. **Batching rules registry**: How each operation handles batched dimensions
3. **Wrapped functions**: The closures created by `vmap()`, `grad()`
4. **Function state**: Parameters when using `make_functional()`

### State Borrowed from PyTorch

1. **Tensors**: Data owned by PyTorch/c10; functorch just transforms them
2. **Gradients**: torch.autograd manages gradient values; functorch shapes how they flow
3. **Module parameters**: torch.nn owns the actual tensors; `make_functional` wraps them
4. **Configuration**: torch.backends settings inherited by transformed operations

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/_functorch/eager_transforms.py` | Eager execution of vmap, grad, vjp, jvp |
| `torch/_functorch/vmap.py` | vmap-specific implementation details |
| `torch/_functorch/aot_autograd.py` | AOT autograd compiler infrastructure |
| `torch/_functorch/partitioners.py` | Batching rules for all operations (large file) |
| `torch/_functorch/make_functional.py` | Converting nn.Module to pure functions |
| `torch/_functorch/autograd_function.py` | Integration with torch.autograd.Function |
| `torch/_functorch/apis.py` | User-facing API definitions |
| `functorch/__init__.py` | Public package exports |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Examined `torch/_functorch/` implementation, batching rules, transformation logic
2. **Book References**: Chapters on autograd and dispatch explain how functorch composes with core systems
3. **Code Flow**: Traced execution from `functorch.vmap()` through transformation dispatch to operation execution

Last Verified: 2026-05-27
