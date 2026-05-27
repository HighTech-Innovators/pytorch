# Architecture Decision Record: torch.jit (JIT Compilation)

## Architectural Role

The `torch.jit` subsystem enables **compilation of PyTorch models** via two mechanisms:

1. **Tracing**: Execute Python code once, record tensor operations into a graph
2. **Scripting**: Parse Python source code, compile to intermediate representation (IR)

Both paths produce a compiled module that can be:
- **Optimized** via IR transformation passes (constant folding, fusion, dead code elimination)
- **Deployed** without a Python interpreter (save/load, ONNX export)
- **Profiled and debugged** at the IR level

Key insight: JIT enables **production deployment and cross-platform inference** by capturing execution as a static graph that can be optimized and run on runtimes without Python.

## Responsibilities

### What This Subsystem Owns

1. **Tracing Infrastructure** (`_trace.py`, C++ `torch/csrc/jit/`)
   - Execute Python function with instrumented tensors
   - Record every operation on traced tensors
   - Build computation graph (IR) from recorded operations
   - Create TracedModule for later execution

2. **Scripting Infrastructure** (`_script.py`, C++ `torch/csrc/jit/`)
   - Parse Python function AST
   - Type inference over AST
   - Generate IR from AST without execution
   - Compile ScriptModule for execution

3. **Intermediate Representation (IR)**
   - Graph-based SSA form representation
   - Operations (aten::, prim::) as nodes
   - Data flow edges between operations
   - Block structure for control flow

4. **Optimization Passes** (C++ `torch/csrc/jit/passes/`)
   - Constant folding: compute constants at compile time
   - Dead code elimination: remove unused operations
   - Common subexpression elimination: merge duplicate operations
   - Inlining: inline function calls
   - Kernel fusion: combine operations into single kernels
   - Type propagation: infer types through graph

5. **Module Classes**
   - `ScriptModule`: Compiled Python module via scripting
   - `TracedModule`: Compiled Python module via tracing
   - Both expose `forward()` method for execution

6. **Debugging and Introspection** (mostly Python)
   - Graph printing and visualization
   - Operation-level debugging
   - Performance profiling of compiled code

### What This Subsystem Does NOT Own

- **Operation Implementations**: ATen defines kernels; JIT only routes to them
- **Memory Management**: c10 allocators handle memory; JIT uses them
- **Automatic Differentiation**: torch.autograd owns gradients; JIT can call autograd but doesn't implement it
- **Python Parsing**: Uses Python's built-in `ast` module
- **Optimization Compilation**: Might use LLVM (external dependency) but not owned here

## Dependencies

### Upstream Dependencies (What Uses This)

- **Model Deployment**: Save compiled modules for inference-only deployment
- **Performance-Critical Code**: JIT compilation enables optimization passes
- **ONNX Export**: Compiled modules converted to ONNX for cross-platform inference
- **Embedded Deployment**: Compiled modules run without Python

### Downstream Dependencies (What This Uses)

- **torch.autograd**: Can trace through backward if autograd enabled
- **ATen**: Operations in IR are ATen operations; execution delegates to ATen
- **c10**: Device and dtype information used in compilation
- **Python ast module**: Standard library for parsing

### Dependency Direction

```
User Code (torch.jit.trace or @torch.jit.script)
    ↓
JIT Compiler (tracing or scripting)
    ↓
IR Generation and Optimization
    ↓
ATen Operation Execution
```

## Trade-offs and Design Decisions

### Tracing vs. Scripting

**Decision**: Support both tracing and scripting; users choose based on code structure.

**Trade-off**:
- ✅ **Advantage**: Tracing is automatic; no annotation burden
- ✅ **Advantage**: Scripting preserves control flow; handles any Python structure
- ❌ **Disadvantage**: Tracing doesn't capture control flow (data-dependent branches fail)
- ❌ **Disadvantage**: Scripting requires learning new language (TorchScript); limited Python support

**Evidence**: Chapter 09 discusses both paths; tracing for dataflow, scripting for complex logic.

### Shape Specialization in Tracing

**Decision**: Traced graphs are specialized to the input shape used during tracing.

**Trade-off**:
- ✅ **Advantage**: Enables shape-specific optimizations
- ✅ **Advantage**: Simpler IR (no shape variables or polymorphism)
- ❌ **Disadvantage**: Tracing with one shape produces graph that fails on other shapes
- ❌ **Disadvantage**: Users must retrace for different batch sizes

**Evidence**: Chapter 09 notes that traced graphs are not shape-polymorphic.

### Graph-Based IR

**Decision**: IR is a DAG of operations, similar to static single assignment form.

**Trade-off**:
- ✅ **Advantage**: Enables global optimization passes (CSE, dead code elimination)
- ✅ **Advantage**: Graph structure easy to visualize and understand
- ❌ **Disadvantage**: Control flow represented as branches; more complex than imperative
- ❌ **Disadvantage**: Loops must be unrolled or handled specially

**Evidence**: IR shown in Chapter 09 uses SSA form with labeled values.

### IR-to-ATen Mapping

**Decision**: IR operations directly map to ATen operations; no intermediate lowering step.

**Trade-off**:
- ✅ **Advantage**: Direct execution; no extra overhead
- ✅ **Advantage**: IR is close to execution level; easier to debug
- ❌ **Disadvantage**: No intermediate representation for alternative backends
- ❌ **Disadvantage**: Harder to target non-ATen backends (requires ONNX export)

**Evidence**: `aten::add`, `aten::matmul`, etc. in IR directly call ATen.

## Extension Boundaries

### Tracing Custom Modules

**Boundary**: Call `torch.jit.trace(module, example_input)` on any `torch.nn.Module`.

```python
model = MyModel()
traced = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
traced.save('model.pt')
```

**Evidence**: Tracing works on any module with `forward()` method.

### Scripting Custom Functions

**Boundary**: Apply `@torch.jit.script` decorator to Python functions with type hints.

```python
@torch.jit.script
def my_function(x: Tensor, y: int) -> Tensor:
    return x * y
```

**Evidence**: Script decorator in `_script.py` compiles decorated functions.

### Custom Optimization Passes

**Boundary**: Add custom IR optimization passes in C++ and register with compiler.

This is advanced; most users don't do this. Requires modifying `torch/csrc/jit/passes/`.

**Evidence**: `torch/csrc/jit/passes/` directory contains pass implementations.

## Runtime Implications

### Lifecycle and Initialization

1. **Definition**: User defines Python module or function
2. **Compilation**: User calls `torch.jit.trace()` or applies `@torch.jit.script` decorator
3. **Execution**: Call compiled module as if it were regular module
4. **Saving**: Call `.save()` to persist compiled module
5. **Loading**: `torch.jit.load()` to load saved module

### Concurrency Behavior

**Thread Safety**:
- **Compilation**: Thread-safe; each trace/script independent
- **Execution**: Compiled modules can be called from multiple threads
- **Module State**: Parameters are shared; in-place modifications not thread-safe

**Evidence**: No explicit locking in compilation; delegates to ATen for thread safety.

### Failure Behavior

1. **Tracing**: If traced with specific shape and called with different shape, may fail or produce incorrect results
2. **Scripting**: Type errors during compilation raise at script time
3. **Unsupported Operations**: If operation not in IR, raises error during compilation or execution
4. **Shape Mismatch**: Dynamic shape inference may fail; runtime shape error

**Evidence**: Tracing failures common when batch size changes; scripting failures at compile time.

## Performance Implications

### Known Hotspots

1. **Compilation Overhead**: Tracing and optimization passes add latency
2. **Graph Execution**: Interpreter-based execution slower than eager (but optimizations may compensate)
3. **Memory Bandwidth**: Optimized graphs may fuse operations, reducing bandwidth

### Allocation Patterns

- **Graph Construction**: Temporary allocations during tracing and optimization
- **Execution**: Operation execution allocates result tensors (same as eager)

### Optimization Benefits

- **Kernel Fusion**: Fused kernels reduce memory bandwidth
- **Constant Folding**: Constants computed once at compile time
- **Dead Code Elimination**: Unused operations removed

## Ownership Boundaries

### State Owned by JIT

1. **Compiled Graph**: IR representation of the computation
2. **Module Metadata**: Module hierarchy, parameter names
3. **Optimization State**: Cached optimized graphs

### State Borrowed from ATen

1. **Operations**: ATen owns operation implementations
2. **Parameter Data**: Tensors own data; JIT only holds references

### State Owned by Users

1. **Original Module**: User-defined PyTorch module
2. **Execution Results**: Outputs produced by executing compiled module

## Key Implementation Files

| File | Purpose |
|---|---|
| `_trace.py` | Tracing infrastructure and API |
| `_script.py` | Scripting infrastructure and API |
| `__init__.py` | JIT module API and exports |
| `csrc/jit/` (C++) | Graph representation, IR, optimization passes |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/jit/` — checked for tracing, scripting, IR structure
2. **Book Chapter**: Chapter 09 "JIT Compilation: From Tracing to Execution" provides architectural understanding
3. **Code Flow**: Traced from user `torch.jit.trace()` call through graph recording and optimization

Last Verified: 2026-05-27
