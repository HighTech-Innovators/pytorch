# Architecture Decision Record: torch._dynamo (Python Bytecode Transformation)

## Architectural Role

`torch._dynamo` is **PyTorch's Python bytecode transformation engine**, capturing Python function execution into a symbolic computation graph. It enables:

1. **Dynamic graph capture**: Converting Python code to FX graphs at runtime
2. **Compilation frontends**: Foundation for torch.compile() and graph capture
3. **Control flow handling**: Supporting Python's if/for/while in compiled code
4. **Variable tracing**: Understanding when variables are symbolic vs. concrete

Key insight: `torch._dynamo` is a **Python-to-graph compiler**. It hooks into Python's execution to observe operations, distinguish tensor operations (symbolic) from Python operations (concrete), and build a graph representation. Unlike static tracing, it executes Python code to handle dynamic control flow.

## Responsibilities

### What This Subsystem Owns

1. **Bytecode Transformation** (`torch/_dynamo/bytecode_transformer.py`)
   - Intercepting Python function execution
   - Converting function bytecode to instrumentation
   - Handling local/global variable scopes

2. **Graph Building** (`torch/_dynamo/guards.py`, `torch/_dynamo/convert_frame.py`)
   - Recording operations as graph nodes
   - Building guards for recompilation (shape changes, type changes)
   - Closure handling

3. **Variable Classification**
   - Determining if variable is tensor (symbolic) or Python scalar (concrete)
   - Tracking variable mutations and dependencies
   - Handling Python objects in tensor computations

4. **Backend Output** (`torch/_dynamo/backends/`)
   - Formatting captured graphs for compiler backends
   - Code generation for compiled execution

5. **Fallback and Debugging** (`torch/_dynamo/logging.py`)
   - Explaining why graph capture failed
   - Debugging tools for graph visualization
   - Resuming to eager execution on unsupported operations

### What This Subsystem Does NOT Own

- **Graph optimization**: torch._inductor owns optimization passes
- **Tensor operations**: ATen owns kernel execution
- **Tracing representation**: torch.fx owns the intermediate representation
- **Automatic differentiation**: torch.autograd owns gradients
- **Compilation backend code generation**: Backend owns this

## Dependencies

### Upstream Dependencies (What Uses This)

- **torch.compile()**: Entry point for compilation
- **torch.fx.symbolic_trace()**: Graph capture for FX tracing
- **Research frameworks**: Custom graph capture implementations

### Downstream Dependencies (What This Uses)

- **torch.fx**: Graph representation and manipulation
- **torch._inductor**: Compilation backend
- **torch.autograd**: For understanding gradient flow
- **sys.settrace()**: Python execution hooking

### Dependency Direction

```
User Code with @torch.compile() or torch.fx.symbolic_trace()
    ↓
torch._dynamo (bytecode transformation)
    ├─→ Bytecode instrumentation
    ├─→ Operation interception
    ├─→ Guard generation
    └─→ torch.fx (graph representation)
        ↓
    torch._inductor (backend compilation)
```

## Trade-offs and Design Decisions

### Execution-Based vs. Static Analysis

**Decision**: Execute Python code to capture graphs (execution-based) rather than static analysis.

**Trade-off**:
- ✅ **Advantage**: Handles dynamic control flow (if/for/while)
- ✅ **Advantage**: Works with Python control flow and side effects
- ❌ **Disadvantage**: Slower graph capture (must execute Python)
- ❌ **Disadvantage**: Requires guards for shape/type changes

**Evidence**: If/for/while handled in `convert_frame.py` via bytecode instrumentation.

### Guards for Recompilation

**Decision**: Generate guards (conditions) for recompiling when graph prerequisites change.

**Trade-off**:
- ✅ **Advantage**: Supports dynamic shapes; recompile only when needed
- ✅ **Advantage**: Handles variable type changes
- ❌ **Disadvantage**: Guard checking overhead
- ❌ **Disadvantage**: Excessive recompilation for highly dynamic code

**Evidence**: `guards.py` defines guard classes; guards checked at compiled code entry.

### Fallback on Unsupported Operations

**Decision**: When an operation isn't supported, resume to eager execution.

**Trade-off**:
- ✅ **Advantage**: Robustness; almost any code can be partially compiled
- ✅ **Advantage**: Incremental optimization possible
- ❌ **Disadvantage**: Unpredictable performance (some ops slower after fallback)
- ❌ **Disadvantage**: Debugging is complex with mixed eager/compiled code

**Evidence**: Fallback logic in `convert_frame.py`; fullgraph mode raises error instead.

## Runtime Implications

### Lifecycle and Initialization

1. **Decoration**: `@torch.compile()` installs bytecode hook
2. **First Call**: Function bytecode transformed; instrumentation added
3. **Execution**: Instrumented function executes; operations recorded
4. **Graph Building**: Captured operations assembled into graph
5. **Backend**: Graph passed to compilation backend
6. **Caching**: Compiled function cached with guard conditions
7. **Subsequent Calls**: Guards checked; use cached version if valid

### Fallback Behavior

- **Unsupported operation**: Operation intercepted, graph dumped, resume to eager
- **Guard failure**: Shape/type changed; rerun capture and recompile
- **Recompilation threshold**: Excessive recompilation triggers fallback warning

## Performance Implications

### Known Hotspots

1. **Bytecode Instrumentation**: Transforming function bytecode on first call
2. **Graph Building**: Recording operations as nodes
3. **Guard Checking**: Checking recompilation conditions at runtime
4. **Excessive Recompilation**: Dynamic code with changing shapes causes repeated compilation

### Optimization Opportunities

- **Guard optimization**: Combine multiple guard checks into single condition
- **Caching strategies**: Reuse compiled code across similar input shapes
- **Eager threshold**: Switch to pure eager for highly dynamic code

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/_dynamo/convert_frame.py` | Main frame/graph conversion logic |
| `torch/_dynamo/bytecode_transformer.py` | Bytecode instrumentation |
| `torch/_dynamo/guards.py` | Guard generation and checking |
| `torch/_dynamo/backends/` | Backend output formats |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `torch/_dynamo/` implementation examined
2. **Book References**: Chapter 9 (JIT Compilation) describes graph capture
3. **Code Flow**: Understanding how bytecode instrumentation works

Last Verified: 2026-05-27
