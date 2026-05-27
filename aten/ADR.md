# Architecture Decision Record: aten (PyTorch Operator Library)

## Architectural Role

The `aten` subsystem is **PyTorch's operator library** — the collection of all tensor operations (1000+) and their implementations. ATen (A Tensor Library) provides:

1. **Operation definitions**: Linear, ReLU, convolution, matrix multiply, etc.
2. **Device-specific implementations**: CPU kernels for each operation
3. **Dispatch logic**: Route operations to appropriate backend implementations
4. **Automatic differentiation integration**: Backward functions for each operation
5. **Code generation infrastructure**: Generate boilerplate from high-level specifications

Key insight: ATen is the **bridge between high-level Python API and low-level CPU/GPU kernels**. Every tensor operation in PyTorch (whether called from Python `torch.add()` or C++ `at::add()`) goes through ATen.

## Responsibilities

### What This Subsystem Owns

1. **Operation Registry** (`aten/src/ATen/native/`)
   - 1000+ operations: add, multiply, convolution, linear, relu, etc.
   - Each operation has a forward implementation (what the operation does)
   - Each operation has backward functions auto-generated for autograd
   - Operations organized in logical groups (math, nn, activation, etc.)

2. **Dispatch System** (`aten/src/ATen/Dispatch.h`, `aten/src/ATen/native/`)
   - Route operations to CPU implementation by device and dtype
   - Fallback mechanisms: use CPU implementation on CPU, CUDA on GPU
   - Quantization dispatch: special paths for quantized tensors
   - Decomposition: expand complex operations into simpler ones

3. **CPU Kernel Implementations** (`aten/src/ATen/native/cpu/`)
   - Actual computation for each operation on CPU
   - Optimized loops, cache-friendly memory access patterns
   - Loop parallelization with OpenMP or other threading
   - SIMD vectorization where applicable

4. **Operator Schema Definition** (defined by torchgen, stored in `aten/src/ATen/native/`)
   - Specifies operation name, arguments, return types
   - Defines backward function signature
   - Enables introspection from Python

5. **Memory and Storage Utilities** (`aten/src/ATen/core/`)
   - TensorAccessor: efficient multidimensional indexing
   - Iterator utilities: parallel iteration over tensor elements
   - Device-specific memory utilities

6. **Functional Interface** (`aten/src/ATen/Tensor.h`, `aten/src/ATen/Functions.h`)
   - `torch.add(a, b)` → pure function
   - `a.add_(b)` → in-place operation
   - Both delegate to same underlying kernel

7. **Backward Function Registration** (`aten/src/ATen/autograd/`, torchgen)
   - Auto-generated backward functions for each operation
   - Specify how gradients flow backward
   - Integration with torch.autograd system

### What This Subsystem Does NOT Own

- **Python Bindings**: torch.csrc defines Python API; ATen is C++ only
- **Automatic Differentiation Orchestration**: torch.autograd handles graph building; ATen only provides backward functions
- **Device Backend Implementation**: NCCL, cuDNN, CUDA kernels are separate (ATen calls them)
- **Memory Allocation**: c10 allocators handle memory; ATen uses them
- **Operator Scheduling**: torch.jit, torch.fx might optimize scheduling; ATen executes operations

## Dependencies

### Upstream Dependencies (What Uses This)

- **torch.autograd**: Reads backward functions from ATen; integrates with backward pass
- **torch.nn**: All layer forward methods call ATen operations
- **torch._C** (Python bindings): Python API routes to ATen through pybind11
- **torch.jit**: Traces or scripts ATen operations for compilation
- **torch.fx**: Captures ATen operations in computational graphs

### Downstream Dependencies (What This Uses)

- **c10**: Uses Device, Tensor, ScalarType, Allocator abstractions
- **BLAS Libraries**: GEMM operations (matrix multiply) call into BLAS (OpenBLAS, MKL)
- **CPU Vendor Libraries**: mkl for Intel CPUs; other optimized libraries
- **External C Libraries**: For specific operations (FFT, etc.)

### Dependency Direction

```
User Code (torch.add, model.forward, etc.)
    ↓
torch.autograd (builds graph, registers backward)
    ↓
ATen Dispatcher (route by device/dtype)
    ↓
ATen CPU Kernels (actual computation)
    ↓
BLAS/System Libraries
```

## Trade-offs and Design Decisions

### Code Generation for Operators

**Decision**: Use torchgen to auto-generate boilerplate (backward functions, schema validation) from high-level specifications.

**Trade-off**:
- ✅ **Advantage**: Reduce boilerplate; forward and backward functions maintain consistency
- ✅ **Advantage**: Easy to add new operations: write template, regenerate code
- ❌ **Disadvantage**: Generated code can be hard to debug
- ❌ **Disadvantage**: Build complexity; code generation step before compilation

**Evidence**: `torchgen/` directory generates `aten/src/ATen/generated/Functions.cpp`.

### Native vs. Composite Operations

**Decision**: Some operations are "native" (lowest-level kernels); others are "composite" (decomposed into other operations).

**Trade-off**:
- ✅ **Advantage**: Composite operations reduce code duplication
- ✅ **Advantage**: Easier to maintain; change implementation in one place
- ❌ **Disadvantage**: Extra function call overhead for composite operations
- ❌ **Disadvantage**: Harder to optimize composite operations as a unit

**Evidence**: `aten/src/ATen/native/` has `cpu/` (native) and other directories; `composite_implicit` and `composite_explicit` operations.

### Dispatch Keys for Layering

**Decision**: Use dispatch keys (CPU, CUDA, Autograd, Quantized, etc.) to layer implementations; operation routed to matching implementation.

**Trade-off**:
- ✅ **Advantage**: Clear separation of concerns; each key handles its aspect
- ✅ **Advantage**: Easy to add new backend (register new dispatch key)
- ❌ **Disadvantage**: Dispatch overhead; hash table lookup for each operation
- ❌ **Disadvantage**: Debugging dispatch chains can be complex

**Evidence**: `aten/src/ATen/Dispatch.h` defines dispatch system; torchgen registers operations per dispatch key.

### In-Place vs. Functional Operations

**Decision**: Support both in-place (e.g., `add_`) and functional (e.g., `add`) versions.

**Trade-off**:
- ✅ **Advantage**: In-place is memory-efficient for iterative algorithms
- ✅ **Advantage**: Functional versions enable immutability for reasoning
- ❌ **Disadvantage**: Maintenance burden; both must be kept in sync
- ❌ **Disadvantage**: Can be source of bugs (in-place side effects)

**Evidence**: `aten/src/ATen/native/` has both `add()` and `add_impl()` patterns.

### Backward Function Registration

**Decision**: Backward functions are auto-generated and registered per operation; autograd invokes them.

**Trade-off**:
- ✅ **Advantage**: Gradient computation correctness guaranteed by generation
- ✅ **Advantage**: Easy to add gradient for new operation
- ❌ **Disadvantage**: Generated code lacks documentation; hard to understand gradient logic
- ❌ **Disadvantage**: Mistakes in generator impact all operations

**Evidence**: `aten/src/ATen/generated/Functions.cpp` contains auto-generated backward functions.

## Extension Boundaries

### Adding New Operations

**Boundary**: Define operation in `aten/src/ATen/native/` with forward and backward logic (or let generator create backward).

1. Add operation definition in yaml/schema format (in torchgen inputs)
2. Implement forward logic in native code
3. Run torchgen to generate backward stub
4. Implement backward logic (or use automatic differentiation)
5. Register operation with dispatcher

**Evidence**: `aten/src/ATen/native/` contains patterns for new operations.

### Adding Backward Support

**Boundary**: Define backward function as static method on operation struct, or rely on auto-generation.

```cpp
// In aten/src/ATen/native/AddOp.cpp
struct AddOp {
  static Tensor forward(Tensor a, Tensor b) {
    return a + b;
  }
  
  static Tensor backward(Tensor grad_output, bool grad_a, bool grad_b) {
    // Backward logic
  }
};
```

**Evidence**: torchgen generates backward functions automatically for simple operations.

### Device-Specific Kernels

**Boundary**: Add device-specific implementation in `aten/src/ATen/<device>/` directory.

1. Add CPU kernel in `aten/src/ATen/native/cpu/`
2. Optionally add CUDA kernel in `aten/src/ATen/native/cuda/` (not present in CPU-only builds)
3. Register with dispatcher for each device

**Evidence**: `aten/src/ATen/native/cpu/`, `aten/src/ATen/native/cuda/` directories.

## Runtime Implications

### Lifecycle and Initialization

1. **Build Time**: torchgen generates operation schemas and backward functions
2. **Compile Time**: Operators compiled into binary; dispatcher initialized
3. **Runtime Startup**: Operation registry populated, dispatch table loaded
4. **Per-Operation**: Dispatch lookup, kernel execution, result return
5. **Backward**: Autograd invokes registered backward functions

### Concurrency Behavior

**Thread Safety**:
- **Operations**: Individual operations are thread-safe for different tensors
- **Shared Tensors**: In-place operations on shared tensors require external synchronization
- **Dispatch**: Dispatcher uses locks for thread-safe registration during startup

**Evidence**: `aten/src/ATen/Dispatch.h` uses static initialization and thread-local storage.

### Failure Behavior

1. **Shape Mismatch**: Operation checks tensor shapes; raises error if incompatible
2. **Dtype Mismatch**: Dispatcher handles dtype; raises error if no compatible kernel
3. **Device Mismatch**: Tensors on different devices; operation raises error
4. **Out of Memory**: Allocator fails; operation propagates OOM error

**Evidence**: ATen kernels contain shape checks and dtype validation.

## Performance Implications

### Known Hotspots

1. **Dispatch Lookup**: Hash table lookup for operation → kernel mapping
2. **BLAS Operations**: Matrix multiply delegates to BLAS (bottleneck for many workloads)
3. **Memory Bandwidth**: Elementwise operations often memory-bound on CPU
4. **Function Call Overhead**: Each operation has function call overhead

### Allocation Patterns

- **Output Tensor Allocation**: Operation allocates result tensor
- **Temporary Buffers**: Some operations allocate scratch space (e.g., reductions)
- **In-Place Reuse**: In-place operations reuse input storage

### Optimization Opportunities

- **Kernel Fusion**: Multiple operations fused into single kernel (handled by torchgen/compiler)
- **Loop Tiling**: Cache-optimal memory access patterns
- **SIMD Vectorization**: Use CPU vector instructions (SSE, AVX, etc.)

## Ownership Boundaries

### State Owned by ATen

1. **Operation Registry**: Dispatcher mapping of operation names to implementations
2. **Backward Function Definitions**: How gradients flow for each operation
3. **Operation Metadata**: Schema (arguments, types, return values)

### State Borrowed from Other Systems

1. **Tensors**: Owned by autograd/torch.nn; ATen only reads/modifies
2. **Memory**: Allocated by c10 allocators; ATen uses
3. **Dispatch Keys**: Provided by c10 and other subsystems

### State Owned by Users

1. **Operation Parameters**: Learning rate, momentum, etc. (if relevant)
2. **Tensor Data**: User owns original tensor; operations produce new tensors

## Key Implementation Files

| File | Purpose |
|---|---|
| `aten/src/ATen/native/` | Native operation implementations |
| `aten/src/ATen/native/cpu/` | CPU-specific kernels |
| `aten/src/ATen/native/cuda/` | CUDA kernels (if CUDA enabled) |
| `aten/src/ATen/Dispatch.h` | Dispatch system infrastructure |
| `aten/src/ATen/TensorMethods.cpp` | Tensor method implementations |
| `aten/src/ATen/Functions.h` | Functional API (torch.add, etc.) |
| `aten/src/ATen/generated/Functions.cpp` | Auto-generated operation signatures |
| `torchgen/` | Code generation for operations and backward functions |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/aten/` — checked for operation definitions, dispatch logic, CPU kernels
2. **Book Chapter**: Chapter 04 "Operators and Dispatch: From Python to CPU" provides architectural understanding
3. **Code Flow**: Traced from high-level operation (e.g., `torch.add()`) through dispatcher to CPU kernel

Last Verified: 2026-05-27
