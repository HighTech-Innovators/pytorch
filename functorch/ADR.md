# Architecture Decision Record: Functorch — Functional Transformations

**Location:** `./src/functorch/`  
**Last Updated:** 2026-05-27  
**Classification:** Extension Layer, Functional Transforms

---

## Executive Summary

Functorch (Functional PyTorch) is a library that extends PyTorch's autograd system with functional transformations: `vmap` (vectorization/batching), `aot_autograd` (ahead-of-time autograd), `eager_transforms` (eager execution transforms), and `make_functional` (functionalizing modules). These transformations enable advanced machine learning techniques like batch processing, tracing, and compilation without modifying user code.

---

## Architectural Role

Functorch serves as the **functional transformation layer** built on top of PyTorch's core. Its responsibilities are:

1. **Vectorization (vmap)** — Transform scalar operation into batched operation without loops
2. **Ahead-of-Time Autograd (aot_autograd)** — Decompose autograd into traceable forward/backward graphs
3. **Eager Transforms** — Apply transformations during eager execution
4. **Functionalization** — Convert stateful modules (nn.Module) into functional operations
5. **Composition** — Enable transformations to be nested (vmap(aot_autograd(...)))
6. **JIT Integration** — Make transformations visible to TorchScript compiler

Because functorch works at the transformation level, **correctness depends on precise control of autograd behavior and graph structure**.

---

## Responsibilities

### What Functorch Does

- **vmap** — Automatically vectorizes operations; user writes for single example, vmap handles batches
- **aot_autograd** — Pre-traces forward and backward graphs; enables compilation and fusion
- **Eager Transforms** — Apply transformations without changing program structure
- **make_functional** — Convert nn.Module to functional form (function + weight tensor)
- **grad_transform** — Explicit gradient transformation (similar to vmap but for gradients)
- **Composition** — Stack transformations (e.g., vmap(aot_autograd(aot_autograd(f))))
- **Tracing Support** — Enable TorchScript to see transformed operations

### What Functorch Does NOT Do

- **Execute Operations** — All actual computation still goes through ATen/autograd
- **Define Operators** — Doesn't register new operations; uses existing ATen ops
- **Allocate Memory** — Memory allocation still handled by tensor creation
- **Modify Autograd Engine** — Works by orchestrating existing autograd, not rewriting it

---

## Dependencies

### What Depends On Functorch

- **torch/compile/** — Uses functorch to decompose operations for compilation
- **torch/fx/** — Tracing framework can use functorch for program transformation
- **Advanced ML Code** — FSDP (fully-sharded data parallel) uses functorch internally
- **User Code** — Any code using vmap, aot_autograd, or functional transformations

### What Functorch Depends On

- **torch/autograd/** — Uses existing autograd system for gradients
- **ATen** — Calls ATen operations for actual computation
- **c10/core/** — Uses types, dispatch system
- **torch/fx/** — Uses for program capture and transformation

**Dependency Direction**: Unidirectional upward. Users and compilation infrastructure depend on functorch; functorch depends downward on autograd and ATen.

---

## Trade-Offs and Design Decisions

### Decision 1: Functional Transforms via Function Composition vs Monolithic Transformation

**What**: Each transformation (vmap, aot_autograd) is a separate function that takes a callable and returns a transformed callable. Transformations compose naturally: `vmap(aot_autograd(f))`.

**Why**:
- Simplicity: Each transformation only knows what it needs to do
- Composability: Transformations can be stacked in any order
- Extensibility: Easy to add new transformations without modifying core
- User Control: User decides which transformations apply

**Alternatives**:
- Monolithic Transformer — single object handles all transformations; harder to extend
- Automatic Transformation Selection — framework chooses transformations; less control

**Trade-offs**:
- **Pro**: Simple, composable, extensible
- **Con**: Composition order matters (vmap(aot_autograd) != aot_autograd(vmap)); user must understand interactions

---

### Decision 2: vmap via Graph Transformation vs Loop Unrolling

**What**: `vmap` works by transforming the computation graph to express batching explicitly, rather than unrolling loops.

**Why**:
- Efficiency: Transformed graph can be compiled and fused
- Generality: Works for any operation that supports batched semantics
- Integration: Works seamlessly with aot_autograd and TorchScript

**Alternatives**:
- Loop Unrolling — user writes for single example; vmap unrolls into loop (slower, limits parallelization)
- Eager Batching — handle batches eagerly in each kernel (doesn't compose well)

**Trade-offs**:
- **Pro**: Efficient, composable, enables compilation
- **Con**: Complex graph transformation logic; harder to debug

---

### Decision 3: aot_autograd via Graph Decomposition

**What**: `aot_autograd` pre-computes forward and backward graphs by symbolic execution, decomposing autograd into an explicit forward() and backward() function.

**Why**:
- Compilation: Explicit backward graph can be compiled to efficient code
- Fusion: Forward and backward can be fused together or independently
- Inspection: Users can see what operations autograd generates
- GPU Efficiency: Backward pass can be optimized separately

**Alternatives**:
- Runtime Autograd — let autograd run normally; no pre-computation (slower for compilation)
- Lazy Backward — defer backward generation until .backward() call (can't fuse beforehand)

**Trade-offs**:
- **Pro**: Enables compilation, fusion, inspection
- **Con**: Complex implementation; decomposition can miss some autograd optimizations

---

### Decision 4: Eager Transforms for Compatibility with Non-Traceable Code

**What**: Transformations work on Python callables, not just traced graphs. Code doesn't need to be TorchScript-compatible.

**Why**:
- Usability: Works with arbitrary Python code including conditionals, loops
- Debugging: Can print and inspect transformations interactively
- Compatibility: Works with existing PyTorch code without modification

**Alternatives**:
- Graph-Only Transforms — require code to be TorchScript-compatible (too restrictive)
- Compile-Required — transformations only available after TorchScript compilation (not flexible)

**Trade-offs**:
- **Pro**: More flexible, easier to debug, works with existing code
- **Con**: Slower than compiled transforms; requires runtime graph construction

---

## Extension Boundaries

### Extending Functorch

**Supported Extensions:**
1. **New Transformation** — Implement new transformation function (e.g., custom sharding transform)
2. **Custom Gradient Computation** — Override how aot_autograd decomposes specific operations
3. **Tracing Integration** — Make new operations visible to vmap/aot_autograd via protocol

**NOT Supported:**
- Modifying autograd behavior (autograd is separate)
- Changing how operations execute (ATen is separate)

### Integration Points

- **torch/autograd/** — Hooks into autograd to capture and replay gradients
- **ATen** — Calls ATen operations for actual computation
- **torch/fx/** — Shares graph representation with FX
- **torch/compile/** — Uses functorch to decompose for compilation

---

## Runtime Implications

### Initialization Order

**At Library Import:**
1. `functorch/_src/__init__.py` — Load transformation implementations
2. `functorch/_src/aot_autograd/` — Initialize ahead-of-time autograd infrastructure
3. `functorch/_src/vmap/` — Initialize vectorization infrastructure

**At First Transform Call:**
- Graph transformation begins
- Autograd hooks registered
- Transformed callable returned

### Concurrency and Thread Safety

- **Transformed Functions** — Thread-safe; each call gets its own graph context
- **Graph Transformation** — Not thread-safe during transformation; transform before multithreading
- **Nested Transforms** — Composable and thread-safe after composition

### Failure Modes

- **Untraceable Code** — Code with operations not supported by vmap will error
- **Gradient Mismatch** — Composed transforms can produce incorrect gradients if interaction not handled (e.g., vmap(aot_autograd) requires special handling)
- **Decomposition Failure** — aot_autograd can fail if backward graph has unsupported operations
- **Infinite Composition** — Deeply nested transforms can cause stack overflow

---

## Performance Implications

### Hot Paths and Allocations

1. **Transformation Overhead** — One-time cost to construct transformed graph
2. **Runtime Dispatch** — Transformed operations dispatch through ATen like normal
3. **Composition Cost** — Each additional transformation adds graph transformation cost

### Known Bottlenecks

- **Graph Transformation Time** — Complex models with many operations take time to transform
- **Memory Usage** — vmap can increase memory usage (multiple batch elements processed)
- **Gradient Computation** — aot_autograd decomposition can be expensive for complex operations
- **Composition Complexity** — Deep nesting of transforms becomes slow

### Mitigation Strategies

- **Caching** — Cache transformed functions to avoid re-transformation
- **Selective Transformation** — Only transform critical code paths
- **Composition Order** — Put faster transforms first to fail fast
- **Eager Compilation** — Use aot_autograd with compilation for best performance

---

## Ownership Boundaries

### What Functorch Owns

- Transformation logic (how to vectorize, how to decompose autograd)
- Graph construction for transformations
- Integration with PyTorch's autograd and dispatch system
- Transformation protocols and interfaces

### What Functorch Does NOT Own

- Autograd implementation (torch/autograd owns this)
- Operation execution (ATen owns this)
- Type system (c10/core owns this)
- Python bindings (torch/ owns this)

### State Shared with Other Layers

- **Transformed Callables** — Functions returned by vmap, aot_autograd
- **Graph Representation** — Shared with torch/fx for tracing
- **Autograd Hooks** — Hooks into existing autograd without modifying it
- **Dispatch System** — Uses existing dispatch; doesn't extend it

---

## Testing and Validation

### Critical Tests

- **Transform Correctness** — Verify vmap produces same results as manual batching
- **Gradient Correctness** — Verify aot_autograd produces correct gradients
- **Composition** — Verify stacked transforms work correctly (vmap(aot_autograd), etc.)
- **Untraceable Code** — Verify error handling for non-transformable operations
- **Performance** — Verify transformations don't significantly slow down execution

### Known Gaps

- Limited testing of deeply nested transformations (>3 levels)
- Limited testing of custom autograd functions with transformations
- No explicit performance regression tests

---

## Related Systems

- **torch/autograd/** — Core autograd that functorch extends
- **ATen** — Actual operation execution
- **torch/fx/** — Program tracing that shares concepts with functorch
- **torch/compile/** — Uses functorch for decomposition
- **torch/nn/** — nn.Module wrapped by make_functional

---

## References

- `functorch/_src/vmap/` — Vectorization implementation
- `functorch/_src/aot_autograd/` — Ahead-of-time autograd implementation
- `functorch/_src/eager_transforms/` — Eager transformation infrastructure
- `functorch/_src/make_functional/` — Module functionalization
- PyTorch GitHub — `pytorch/functorch` repository (open source)
