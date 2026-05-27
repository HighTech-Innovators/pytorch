# Architecture Decision Record: torch.compiler (Compilation Infrastructure)

## Architectural Role

`torch.compiler` is **PyTorch's compilation frontend**, providing `torch.compile()` decorator and infrastructure for transforming Python functions into optimized executables. It enables:

1. **Graph-based optimization**: Capturing computation as a graph for global optimization
2. **Backend selection**: Supporting multiple compiler backends (TorchInductor, ONNX, etc.)
3. **Performance improvement**: Trading compilation time for faster execution
4. **Debugging support**: Tracing and visualization of compiled code

Key insight: `torch.compiler` is a **bridge between eager execution and compiled code**. It captures the execution graph dynamically (via TorchDynamo Python bytecode transformation), optimizes it, and delegates to a backend compiler. The compilation is transparent to user code.

## Responsibilities

### What This Subsystem Owns

1. **Public Compilation API** (`torch.compile()`)
   - User-facing decorator for function/module compilation
   - Options: backend selection, fullgraph mode, dynamic shapes, etc.
   - Error handling and fallback behavior

2. **Backend Abstraction** (`torch.compiler.BackendWrapper`)
   - Interface for compiler backends
   - Backend registration and selection
   - Protocol for calling into compilation

3. **Compilation Configuration** (`torch.compiler.config`)
   - Global compilation settings
   - Per-compilation options
   - Debugging and tracing controls

4. **Graphviz Visualization** (`torch.compiler.draw_graph`)
   - Visualize captured computation graphs
   - Used for debugging and understanding compiled code

### What This Subsystem Does NOT Own

- **Graph capture**: torch._dynamo owns bytecode transformation
- **Graph optimization**: TorchInductor or backend owns optimization
- **Kernel execution**: Backend compilers and ATen own execution
- **Automatic differentiation**: torch.autograd owns gradient computation
- **Tracing infrastructure**: torch.fx and torch._inductor own tracing

## Dependencies

### Upstream Dependencies (What Uses This)

- **User code**: Any Python code decorated with `@torch.compile()`
- **Model training pipelines**: Performance-critical paths
- **Framework integration**: Libraries building on PyTorch

### Downstream Dependencies (What This Uses)

- **torch._dynamo**: Python bytecode transformation and graph capture
- **torch._inductor**: Default compilation backend
- **torch.fx**: Graph representation and transformations
- **torch.onnx**: ONNX export backend
- **torch.autograd**: Gradient computation on compiled code

### Dependency Direction

```
User Code with @torch.compile()
    ↓
torch.compiler (frontend)
    ├─→ torch._dynamo (graph capture)
    ├─→ Backend selection
    └─→ torch._inductor (default backend)
        ├─→ torch.fx (graph representation)
        ├─→ Optimization passes
        └─→ Code generation
```

## Trade-offs and Design Decisions

### Single Decorator for Multiple Backends

**Decision**: Single `@torch.compile()` works with multiple backends (TorchInductor, ONNX, etc.).

**Trade-off**:
- ✅ **Advantage**: Unified API; users don't need to learn backend-specific APIs
- ✅ **Advantage**: Easy to switch backends without code changes
- ❌ **Disadvantage**: Backend-specific options surface through generic `options` dict
- ❌ **Disadvantage**: Error messages and behavior differ across backends

**Evidence**: `torch.compile(backend="inductor")` vs `torch.compile(backend="onnx")`.

### Dynamic Compilation on First Call

**Decision**: Capture and compile graph on first execution, cache for subsequent calls.

**Trade-off**:
- ✅ **Advantage**: Transparent to user code; no explicit compilation step
- ✅ **Advantage**: Works with dynamic shapes and control flow
- ❌ **Disadvantage**: First call has high latency (compilation overhead)
- ❌ **Disadvantage**: Slower startup time for short-running scripts

**Evidence**: Compilation happens inside wrapped function; subsequent calls use cached compiled code.

### Fullgraph vs. Fallback Mode

**Decision**: Offer two modes: `fullgraph=True` (fail if can't compile entire graph) or fallback (switch back to eager on unsupported ops).

**Trade-off**:
- ✅ **Advantage**: Fullgraph ensures predictable performance; fallback provides robustness
- ✅ **Advantage**: Users can choose tradeoff between safety and performance
- ❌ **Disadvantage**: Two modes to understand and debug
- ❌ **Disadvantage**: Fallback mode may hide compilation issues

**Evidence**: `torch.compile(fullgraph=True)` for strict mode.

## Runtime Implications

### Lifecycle and Initialization

1. **Decoration**: `@torch.compile()` wraps function; no compilation yet
2. **First Call**: Function executed normally; TorchDynamo captures graph
3. **Graph Transformation**: Captured graph optimized by backend
4. **Compilation**: Backend compiles optimized graph to executable
5. **Caching**: Compiled code cached for reuse
6. **Subsequent Calls**: Use cached compiled code

### Fallback Behavior

- **Unsupported operation**: Falls back to eager execution if `fullgraph=False`
- **Dynamic shapes**: TorchDynamo reruns capture for each unique shape
- **Control flow**: Complex Python control flow may trigger recompilation

## Performance Implications

### Known Hotspots

1. **Graph Capture**: TorchDynamo bytecode transformation on first call
2. **Compilation**: Backend compilation time (can be seconds for large models)
3. **Overhead at boundary**: Switching between eager and compiled code

### Allocation Patterns

- **Compilation artifacts**: Compiled code and optimization data cached in memory
- **Graph buffers**: Intermediate representations during compilation
- **Kernel specialization**: Per-shape compiled code variants

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/compiler/__init__.py` | Public API and registration |
| `torch/compiler/blackhole.py` | No-op backend (for testing) |
| `torch/compiler/inductor.py` | Integration with TorchInductor backend |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `torch/compiler/` module structure
2. **Code Flow**: Understanding of how `@torch.compile()` decorator works
3. **Integration**: Interaction with torch._dynamo, torch._inductor

Last Verified: 2026-05-27
