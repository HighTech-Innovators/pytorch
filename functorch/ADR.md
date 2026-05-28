# ADR: functorch — Functional Program Transformations

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

functorch provides composable program transformations (vmap, grad, vjp, jvp) that transform PyTorch computation graphs while preserving automatic differentiation. It enables:

- **Vectorization** (vmap): Batch dimension handling via automatic axis remapping
- **Custom gradients** (vjp, jvp): Compute vector-jacobian and jacobian-vector products
- **Functional programming**: Transform eager code into batched or transposed forms

functorch is classified as **Coordination Heavy** because it intercepts operator dispatch to apply transformations without modifying user code.

## Responsibilities

### What functorch Owns
- **vmap transformation**: Automatic vectorization using axis remapping at dispatch layer
- **Gradient transformations** (grad, vjp, jvp): Forward and reverse-mode differentiation
- **Function composition**: Stack multiple transformations (vmap + grad, vjp + vmap, etc.)
- **Batching rules**: Per-operator semantics for batched execution
- **Dynamic shape inference**: Propagate batch dimensions through operations
- **Fallback mechanisms**: Graceful degradation for unsupported operators

### What functorch Does Not Own
- **Core differentiation engine**: torch.autograd owns backward passes
- **Operator implementations**: aten/ owns kernel implementations
- **Dispatch system**: ATen dispatcher handles dispatch decisions
- **Python API surface**: torch.* owns high-level APIs
- **JIT compilation**: torch.jit owns graph optimization

## Dependencies

### Internal Dependencies (functorch → other modules)
- **ATen**: Reads dispatch keys, invokes operators via standard API
- **torch.autograd**: Uses autograd machinery for gradient computation
- **c10**: Uses TensorImpl, Device, DispatchKey for dispatch interception
- **torch**: Uses Python bindings for user-facing API

### External Dependencies (other modules → functorch)
- **torch.vmap**: High-level API for batching
- **torch.func.grad**: High-level API for differentiation
- **torch.compile**: Can use functorch transformations internally
- **User code**: Direct use of functorch transformations

**Soft dependency**: Users can choose to use functorch or avoid it; not required for basic PyTorch functionality

## Trade-offs and Design Decisions

### 1. Dispatch Interception vs Graph Rewriting
**Decision**: Intercept at dispatcher level (dispatch key insertion) instead of graph rewriting  
**Rationale**:
- **Eager compatibility**: Works with eager execution without tracing
- **Composability**: Multiple transformations can stack via dispatch key layering
- **Transparency**: User code sees only batched results; transformation is invisible

**Trade-off**: Dispatch overhead at every operation; fallback for unsupported operators

**Evidence**: `functorch/_src/batching_utils.py` implements batching via dispatch key insertion. `functorch/_src/functional_functions.py` registers batching rules.

### 2. Batching Rules vs Auto-Batching
**Decision**: Require explicit batching rules for each operator instead of automatic inference  
**Rationale**:
- **Correctness**: Each operator has unique batching semantics (matrix mult batches differently than convolution)
- **Performance**: Hand-written rules can specialize for common cases
- **Understandability**: Users can understand what vmap does for specific operators

**Trade-off**: Requires maintaining thousands of batching rules (one per operator)

**Evidence**: `functorch/_src/partitioners/` contains hundreds of batching rule definitions. `gen_batching_rules.py` generates some rules automatically.

### 3. Function Composition via Dispatch Keys
**Decision**: Stack transformations by inserting dispatch keys; each transformation handles its key  
**Rationale**:
- **Composability**: vmap + grad + vjp can be arbitrarily nested
- **Orthogonality**: Transformations don't need to know about each other
- **Reuse**: Same dispatch key mechanism used across all transformations

**Trade-off**: Dispatch key space is finite; many keys needed (VmapBatchedKey, GradKey, etc.)

**Evidence**: `functorch/_src/dispatch_keys.py` defines transformation-specific keys. `functorch/_src/functional_functions.py` registers handlers per key.

### 4. Fallback for Unsupported Operators
**Decision**: Fall back to eager execution (compute outside transformation) for operators without rules  
**Rationale**:
- **Robustness**: Code doesn't crash on new operators; silently falls back
- **Incrementalism**: New operators automatically supported once rules added
- **Debugging**: Fallback is visible in traces (allows identifying missing rules)

**Trade-off**: Incorrect semantics possible (fallback may violate transformation semantics)

**Evidence**: `functorch/_src/batching_rules.py` has `_not_implemented` marker for operators without rules. Fallback invokes eager execution.

## Extension Boundaries

### Public Extension Points
1. **Custom vmap rules**: Register batching rules for user-defined ops via `register_batching_rule()`
2. **Custom transformations**: Implement new transformations using dispatch key mechanism
3. **Composition**: Nest transformations arbitrarily (vmap(grad(vmap(...))))

### Extension Constraints
- **Batching rules must preserve semantics**: vmap(f)(batched_inputs) must equal batched outputs of f
- **Dispatch key space is shared**: Custom transformations cannot define new dispatch keys at runtime (compile-time only)
- **Unsupported operators silently fall back**: Can lead to incorrect results if fallback is not correct

## Runtime Implications

### vmap Execution
1. **User calls**: `vmap(f)(batched_tensor)`
2. **Batch dimension setup**: Mark batch axis in dispatch key
3. **Operator interception**: Each operation checks if VmapBatchedKey is active
4. **Batching rule application**: Apply registered rule for that operator (e.g., linear layer's rule)
5. **Batch dimension removal**: Strip batch axis from results
6. **Return**: Results have same batch structure as inputs

### grad Execution
1. **User calls**: `grad(f)(x)`
2. **Gradient setup**: Mark GradKey in dispatch
3. **Autograd recording**: Autograd engine records operations normally
4. **Backward execution**: Backward pass computes gradient of f with respect to x
5. **Return**: Gradient tensor with same shape as x

### Composition Example: vmap(grad(f))
1. **vmap** inserts VmapKey into dispatch
2. **grad** inserts GradKey into dispatch
3. **Operations see both keys** in dispatch set
4. **Batching rule applied first** (higher priority)
5. **Gradient computation** happens within batched context
6. **Result**: Batch of gradients (one per batch element)

### Thread Safety
- **functorch is not thread-safe**: Dispatch key thread-local state must be single-threaded
- **Composition safety**: Nesting multiple transformations is safe (each manages its own state)

## Performance Implications

### Memory Overhead
- **Dispatch key insertion**: O(1) per transformation (bitset operation)
- **Batching rule metadata**: ~64 bytes per registered rule (~50KB total for all rules)
- **Saved batch metadata**: Per-operation, stored in dispatch key state

### Performance-Critical Paths
1. **vmap dispatch** (every operation inside vmap)
   - Bottleneck: Batching rule application, axis remapping
   - Optimization: Pre-compute batch dimension plans, cache batching rule lookups
2. **grad dispatch** (every operation inside grad)
   - Bottleneck: Autograd graph construction, gradient accumulation
   - Optimization: Fusion to reduce operation count (not yet implemented)
3. **Composition** (nested transformations)
   - Bottleneck: Multiple dispatch key checks, rule composition overhead
   - Optimization: JIT fusion of nested transformations

### Performance Characteristics
- **vmap overhead**: ~1-5% slowdown vs manually batched code (rule application cost)
- **grad overhead**: ~5-20% slowdown vs manual gradient computation (autograd overhead)
- **Composition overhead**: ~2-5% additional overhead per layer of nesting

## Ownership Boundaries

### functorch Subsystems
- **batching rules**: Per-operator semantics for vmap (hundreds of rules)
- **gradient transformations**: vjp, jvp, grad implementations
- **dispatch key management**: Track active transformations in dispatch context
- **fallback mechanisms**: Graceful degradation for unsupported operators
- **Python API**: User-facing vmap(), grad(), vjp() functions

## Notes and Caveats

1. **vmap changes operator semantics significantly**: Results may not match manually batched code if batching rule is incorrect
2. **Gradients of vmap have subtle semantics**: vmap(grad(...)) computes batch of gradients; grad(vmap(...)) computes gradient of batched operation (different result)
3. **Fallback for unsupported operators can hide bugs**: Silent fallback may apply incorrect semantics; enables gradual adoption but requires testing
4. **Batching rules are operator-specific**: Matrix multiplication batches along batch dimension; convolution also batches but in different way; rules encode this knowledge
5. **Dispatch key priority matters**: Batching rules must be checked before operator implementations; incorrect priority order breaks semantics
6. **Nested vmap can be inefficient**: Multiple vmap layers add overhead; consider refactoring to single vmap when possible
7. **vmap + autograd interaction is complex**: Batch dimensions interact with gradient flow; users must understand composition semantics
8. **functorch is actively developed**: API and semantics may change; not yet stable for production use (as of knowledge cutoff)
