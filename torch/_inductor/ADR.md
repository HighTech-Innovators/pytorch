# Architecture Decision Record: torch._inductor (Compilation Backend)

## Architectural Role

`torch._inductor` is **PyTorch's graph compilation backend**, converting captured computation graphs into optimized C++/Triton code. It enables:

1. **Kernel fusion**: Combining multiple operations into single kernel
2. **Memory optimization**: Reordering computations to improve cache locality
3. **Loop tiling**: Optimizing loop structures for hardware
4. **Code generation**: Producing C++/Triton kernels from graph
5. **Dynamic shapes**: Supporting variable tensor dimensions at runtime

Key insight: `torch._inductor` is the **optimizer and code generator** that turns high-level computation graphs (from torch._dynamo) into low-level, optimized kernels. It's the compilation backend that makes `torch.compile()` achieve performance.

## Responsibilities

### What This Subsystem Owns

1. **Graph Analysis** (`torch/_inductor/scheduler.py`)
   - Analyzing dependencies between operations
   - Identifying fusion opportunities
   - Computing operation ordering

2. **IR Generation** (`torch/_inductor/ir.py`)
   - Building intermediate representation from FX graphs
   - Lowering high-level operations to IR primitives
   - IR nodes for computation and memory operations

3. **Kernel Fusion** (`torch/_inductor/fusion.py`)
   - Identifying operations that can be combined
   - Merging operations into single kernel
   - Loop fusion strategies

4. **Memory Optimization** (`torch/_inductor/memory_planning.py`)
   - Allocating memory for intermediate tensors
   - Reusing memory when possible
   - Minimizing memory footprint

5. **Code Generation**
   - Generating C++ or Triton code from IR
   - Loop generation and tiling strategies
   - Register allocation

6. **Device Backend Selection**
   - Supporting CPU, CUDA, other devices
   - Backend-specific code generation
   - Device capability queries

### What This Subsystem Does NOT Own

- **Graph capture**: torch._dynamo owns bytecode transformation
- **Tensor operations**: ATen owns operation implementations
- **Device management**: torch.device and backends own this
- **Automatic differentiation**: torch.autograd owns gradient computation
- **Execution**: C++ runtime and Triton compiler execute generated code

## Dependencies

### Upstream Dependencies (What Uses This)

- **torch.compile()**: Default backend for compilation
- **torch._dynamo**: Passes captured graphs to inductor
- **Research frameworks**: Custom compilation frontends

### Downstream Dependencies (What This Uses)

- **torch.fx**: Graph representation
- **Triton compiler**: For Triton code generation
- **C++ compiler**: For C++ code generation
- **torch**: For tensor and operation definitions

### Dependency Direction

```
torch._dynamo (captured graph)
    ↓
torch._inductor (compilation backend)
    ├─→ IR generation
    ├─→ Graph analysis & scheduling
    ├─→ Kernel fusion
    ├─→ Memory optimization
    ├─→ Code generation
    └─→ C++/Triton compilation
```

## Trade-offs and Design Decisions

### Greedy Scheduling vs. Optimal

**Decision**: Use greedy scheduling heuristics rather than exact optimization.

**Trade-off**:
- ✅ **Advantage**: Fast compilation; can generate code in seconds
- ✅ **Advantage**: Scales to large graphs
- ❌ **Disadvantage**: Generated code may not be optimal
- ❌ **Disadvantage**: Some fusions missed due to heuristics

**Evidence**: Scheduler uses greedy algorithms for fusion and scheduling.

### Single Backend (Default to C++)

**Decision**: Default to C++ code generation; Triton as alternative for GPU.

**Trade-off**:
- ✅ **Advantage**: Portable; C++ runs on all platforms
- ✅ **Advantage**: Familiar toolchain and debugging
- ❌ **Disadvantage**: C++ may not be optimal for all hardware
- ❌ **Disadvantage**: Compilation time for C++ can be slower than interpreted IR

**Evidence**: Code generation in `torch/_inductor/` targets C++.

### Compile-Time Shape Specialization

**Decision**: Generate specialized code for specific input shapes; recompile for new shapes.

**Trade-off**:
- ✅ **Advantage**: Generated code can use concrete shapes for optimization
- ✅ **Advantage**: Better performance for fixed shapes
- ❌ **Disadvantage**: Recompilation overhead for dynamic shapes
- ❌ **Disadvantage**: Large code cache for many shape variants

**Evidence**: Inductor generates per-shape compiled kernels.

## Runtime Implications

### Lifecycle and Initialization

1. **Graph capture**: torch._dynamo captures FX graph
2. **Inductor initialization**: Backend selected, configuration loaded
3. **IR generation**: FX graph converted to IR
4. **Scheduling**: Operations ordered and dependencies computed
5. **Fusion**: Identify and merge operations
6. **Code generation**: Generate C++/Triton
7. **Compilation**: C++ or Triton compiler compiles to binary
8. **Caching**: Compiled code cached for reuse
9. **Execution**: Invoke compiled code

### Fallback Behavior

- **Unsupported operation**: Raise error or fall back to eager
- **Compilation failure**: Revert to eager execution
- **Code size explosion**: May generate very large code for complex graphs

## Performance Implications

### Known Hotspots

1. **Scheduling**: Computing optimal operation order can be expensive
2. **Fusion Analysis**: Checking all pairs of operations for fusion
3. **Code generation**: Generating C++ or Triton code
4. **Compilation**: C++ compiler time (potentially seconds)

### Optimization Opportunities

- **Caching compiled code**: Reuse compiled kernels across runs
- **Incremental compilation**: Compile subgraphs in parallel
- **Hybrid backends**: Mix C++ and Triton based on operation type

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/_inductor/scheduler.py` | Graph scheduling and fusion |
| `torch/_inductor/ir.py` | Intermediate representation |
| `torch/_inductor/codegen/` | Code generation for backends |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Examined `torch/_inductor/` structure
2. **Code Flow**: Understanding compilation pipeline from graph to code
3. **Integration**: How inductor receives graphs from dynamo

Last Verified: 2026-05-27
