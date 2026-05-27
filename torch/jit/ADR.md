# Architecture Decision Record: JIT Compiler and TorchScript (torch.jit)

## Architectural Role

**torch.jit** is PyTorch's just-in-time (JIT) compiler and TorchScript system — the machinery for converting Python models into optimized, serializable representations that can run without Python. It provides two compilation modes: tracing (record operations during forward pass) and scripting (parse Python code into TorchScript IR). This enables model deployment, performance optimization, and mobile inference.

**Location**: `torch/jit/`, `torch/csrc/jit/` | **Language**: Python frontend, C++ backend | **Dependencies**: ATen, autograd, torch.nn

## Responsibilities

**torch.jit owns**:
- Tracing mode (`torch.jit.trace()`): record forward pass and build operation graph
- Scripting mode (`torch.jit.script()`): parse Python AST and compile to IR
- TorchScript IR representation and graph optimization
- ScriptModule class for encapsulating traced/scripted models
- Model serialization and deserialization (`torch.jit.save()`, `torch.jit.load()`)
- Graph optimization passes (constant folding, dead code elimination, operator fusion)
- Type inference and checking for scripted code
- Runtime execution engine for TorchScript

**torch.jit does not own**:
- Tensor operations (ATen owns those)
- Gradient computation (autograd owns that)
- Python parsing beyond AST (Python standard library owns that)
- ONNX export (torch.onnx owns that)

## Dependencies

### Inbound Dependencies
- **User code** calls `torch.jit.trace()` or `@torch.jit.script` decorator
- **Model deployment** loads TorchScript via C++ API
- **Mobile inference** loads serialized TorchScript models
- **Performance optimization** uses JIT to compile models

### Outbound Dependencies
- **ATen** for tensor operations in TorchScript IR
- **autograd** for automatic differentiation (if model has gradients enabled)
- **torch.nn** for module tracing

## Trade-offs and Design Decisions

### 1. Two Compilation Modes: Tracing vs Scripting
**Decision**: Provide both tracing (record-replay) and scripting (parse AST) modes.

**Rationale**: 
- Tracing: simple, works with any Python code (including C++ extensions), no type annotations needed
- Scripting: preserves control flow (if/while), type-safe, but requires subset of Python
- Different use cases: tracing for data-parallel models, scripting for control-flow-heavy models

**Alternative**: Only support one mode (simpler but less flexible)

**Trade-off**: Users must choose mode; tracing has control flow limitations, scripting requires type annotations.

### 2. TorchScript IR as Graph of Operations
**Decision**: Compile to graph-based intermediate representation (IR) where each node represents ATen operation.

**Rationale**: 
- Enables optimization passes: fuse operations, eliminate dead code, constant fold
- Graph serialization: can serialize and load in C++ without Python
- Cross-platform: IR is independent of Python version

**Evidence**: torch/csrc/jit/ir/ir.h defines graph structure.

**Trade-off**: Graph IR is not human-readable; harder to debug than Python.

### 3. Type Inference for Scripted Code
**Decision**: Scripting mode includes type inference to infer types of variables without explicit annotations.

**Rationale**: 
- Python is dynamically typed, but TorchScript IR needs static types
- Type inference allows idiomatic Python code in many cases
- Reduces annotation burden on users

**Trade-off**: Type inference can fail on ambiguous code; user must add explicit type hints.

### 4. ScriptModule as Serializable Wrapper
**Decision**: Both traced and scripted models wrapped in `ScriptModule` class with `.save()` and `.load()` methods.

**Rationale**: 
- Uniform interface for both tracing and scripting modes
- `ScriptModule` can be saved to .pt file (pickle format with torch metadata)
- Loaded in C++ without Python runtime

**Evidence**: ScriptModule in torch/jit/__init__.py

### 5. Operator Fusion Optimization
**Decision**: Optimization pass fuses multiple operations into single fused kernel (e.g., matmul + add → specialized kernel).

**Rationale**: 
- Reduces memory bandwidth by fusing elementwise operations
- Reduces kernel launch overhead
- Improves CPU cache locality

**Trade-off**: Fusion adds compilation time; only beneficial for frequently-called models.

### 6. Constant Folding Optimization
**Decision**: Optimization pass evaluates operations on compile-time constants at compilation time.

**Rationale**: 
- Reduces runtime computation: `torch.tensor([1,2,3]) + 1` → compile to constant tensor
- Reduces model size and memory footprint
- No runtime cost

**Trade-off**: Requires value tracking at compile time; some operations may not be foldable.

### 7. Python Subset for Scripting
**Decision**: Scripting supports only a subset of Python: control flow, basic types, functions, classes. No stdlib, no dynamic features.

**Rationale**: 
- Need to guarantee compilation to static IR
- Avoid Python stdlib complexity
- Enable C++ code generation

**Alternative**: Support full Python (impossible; too complex)

**Trade-off**: Users cannot use arbitrary Python libraries inside scripted functions.

### 8. Graph Serialization Format
**Decision**: TorchScript models saved in .pt format (Python pickle with custom torch extensions).

**Rationale**: 
- Pickle format widely used in Python ecosystem
- Custom extensions handle torch-specific types (Tensor, Device, etc.)
- Can be loaded in C++ via libtorch

**Trade-off**: .pt files contain Python bytecode (potential security risk); should not load from untrusted sources.

## Extension Boundaries

**Custom operators**: Register custom operators via `torch.library` or `torch.ops.<namespace>.<op>`; available in TorchScript.

**Optimization passes**: Add custom graph optimization passes to JIT compiler.

**Serialization formats**: ONNX export (torch.onnx) converts TorchScript IR to ONNX format.

**Mobile models**: Optimize TorchScript models for mobile (torch.jit.optimized_for_mobile()).

## Runtime Implications

### Initialization
- JIT compiler initialization: load at import time (torch/jit/__init__.py)
- Tracing setup: minimal overhead
- Scripting setup: Python AST parser loaded

### Tracing Process
1. User calls `torch.jit.trace(model, example_input)`
2. Record mode enabled: operations execute normally but also record in operation graph
3. Forward pass with example input
4. Build graph from recorded operations
5. Return ScriptModule wrapping graph

**Overhead**: ~10-100ms depending on model size; recorded graph smaller than Python code.

### Scripting Process
1. User decorates function with `@torch.jit.script` or calls `torch.jit.script(func)`
2. Parse Python AST
3. Type inference on AST
4. Compile to TorchScript IR
5. Return ScriptModule

**Overhead**: ~100ms-1s depending on function complexity; one-time at definition time.

### Execution
- ScriptModule forward pass: execute IR (interp or compiled)
- No Python overhead: runs directly on C++ IR executor
- Same ATen kernels as Python path, but no dispatch overhead

### Memory
- Traced model: graph ~10-30% size of original Python code
- Scripted model: IR optimized further, often smaller than traced
- Serialized .pt file: model size plus metadata

### Concurrency
- **Not thread-safe** for concurrent tracing or scripting compilation
- **Safe** for concurrent forward passes on same ScriptModule (read-only after compilation)
- Concurrent execution on different ScriptModules on different devices is safe

### Lifecycle
- ScriptModule immutable after compilation
- Can be serialized and deserialized repeatedly
- No state modification (weights fixed at compile time)

## Performance Implications

### Known Hotspots
1. **Tracing overhead**: Recording operations adds ~5-10% overhead to forward pass
2. **Compilation overhead**: Scripting compilation is slow; one-time cost
3. **Graph execution**: Interpreter-based execution has overhead vs compiled

### Optimization Opportunities
1. **Operator fusion**: Fuse elementwise operations into single kernel
2. **Constant folding**: Evaluate constant expressions at compile time
3. **Dead code elimination**: Remove unused operations
4. **Allocator optimization**: Pool allocations for intermediate tensors
5. **Mobile optimization**: torch.jit.optimized_for_mobile() removes operators not available on mobile

### Deployment Performance
- C++ execution without Python: 1-10x faster than Python for small models
- Larger models: speedup depends on computation vs Python interpretation overhead
- Mobile: major optimization opportunity; can reduce model size 10-100x

## Ownership Boundaries

**torch.jit owns**:
- Tracing and scripting mechanisms
- TorchScript IR and optimization
- ScriptModule class and serialization
- Type inference for scripted code

**torch.jit delegates to**:
- ATen for tensor operations in executed IR
- autograd for gradient computation (if enabled)
- Python AST parser for scripting front end

**Related systems own**:
- torch.onnx (ONNX export from TorchScript)
- torch.utils.data (data loading for tracing)
- C++ libtorch API (loading and executing TorchScript in C++)
- torch.jit.quantization (quantizing TorchScript models)
