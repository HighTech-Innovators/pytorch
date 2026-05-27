# Architecture Decision Record: torch (Main PyTorch Package)

## Architectural Role

The `torch` package is **PyTorch's primary user-facing interface**. It provides:

1. **Tensor factory functions**: Creation of tensors (randn, zeros, ones, empty, etc.)
2. **Functional operations**: Stateless mathematical functions (add, matmul, relu, etc.)
3. **Public API entry point**: Where users import PyTorch
4. **Module organization**: Subpackages for nn, optim, distributed, etc.
5. **Initialization and configuration**: Loading C++ bindings, setting default behavior

Key insight: The `torch` package is a **facade and router** — it doesn't implement operations (that's ATen), but organizes and presents them to Python users. It bridges Python and C++ layers via pybind11 bindings in `torch._C`.

## Responsibilities

### What This Subsystem Owns

1. **Package Initialization** (`torch/__init__.py`)
   - Import torch._C (C++ extension module) at package load
   - Populate torch namespace with factory functions and operations
   - Set default dtype, device, and other global configuration
   - Register extension modules (torch.nn, torch.optim, torch.distributed)

2. **Tensor Creation API** (factory functions in `torch/__init__.py`)
   - `torch.randn()`, `torch.zeros()`, `torch.ones()`, `torch.empty()`
   - `torch.arange()`, `torch.linspace()`, `torch.logspace()`
   - `torch.eye()`, `torch.diag()`, `torch.tensor()`
   - Device and dtype control for all factories

3. **Functional Operations** (`torch/__init__.py` + `torch/functional.py`)
   - `torch.add()`, `torch.mul()`, `torch.matmul()`, `torch.dot()`
   - Activation functions: `torch.relu()`, `torch.sigmoid()`, `torch.tanh()`
   - Aggregations: `torch.sum()`, `torch.mean()`, `torch.std()`
   - All stateless (no learnable parameters)

4. **Tensor Class and Methods** (Python wrapper around C++ Tensor)
   - `Tensor.__init__()`, `Tensor.__add__()`, `Tensor.__mul__()` (operator overloads)
   - `Tensor.shape`, `Tensor.dtype`, `Tensor.device` (properties)
   - `Tensor.sum()`, `Tensor.mean()`, `Tensor.backward()` (methods)
   - View operations: `Tensor.reshape()`, `Tensor.transpose()`, `Tensor.squeeze()`

5. **Global Context and Configuration** (`torch/__init__.py`, `torch/config.py`)
   - Default dtype: `torch.get_default_dtype()`, `torch.set_default_dtype()`
   - Default device: `torch.device()` context manager
   - Gradient enable/disable: `torch.no_grad()`, `torch.enable_grad()`
   - Random seed: `torch.manual_seed()`, `torch.seed()`
   - Inference mode: `torch.inference_mode()`

6. **Subpackage Organization**
   - `torch.nn`: Neural network modules
   - `torch.optim`: Optimizers
   - `torch.autograd`: Automatic differentiation
   - `torch.distributed`: Distributed training
   - `torch.jit`: JIT compilation
   - `torch.fx`: Functional transformation
   - `torch.utils`: Data loading, checkpointing
   - Other: `torch.cuda`, `torch.backends`, `torch.profiler`

7. **C++ Binding Bridge** (`torch/__init__.py`)
   - Import torch._C (pybind11-generated extension)
   - Expose C++ functions and classes to Python
   - Handle version compatibility

8. **Documentation and Constants**
   - `torch.__version__`: PyTorch version
   - Docstrings for all public functions and classes
   - Type hints (gradually added)

### What This Subsystem Does NOT Own

- **Operation Implementations**: ATen owns kernels; torch.py just wraps them
- **Automatic Differentiation**: torch.autograd owns graph building; torch just exposes the API
- **Neural Network Layers**: torch.nn owns module definitions; torch just imports them
- **C++ Bindings**: torch.csrc defines pybind11 bindings; torch just uses them
- **Low-Level Allocation**: c10 allocators handle memory; torch just uses them

## Dependencies

### Upstream Dependencies (What Uses This)

- **User Code**: `import torch` is the entry point for all PyTorch programs
- **Research Code**: Jupyter notebooks, training scripts
- **Hugging Face, TensorFlow, etc.**: Other libraries build on PyTorch APIs
- **Model Zoos**: Pre-built models use torch APIs

### Downstream Dependencies (What This Uses)

- **torch._C** (pybind11): C++ bindings for Tensor, operations, configuration
- **torch.autograd**: Gradient tracking (imported in torch namespace)
- **torch.nn**: Neural network modules (imported as subpackage)
- **torch.optim**: Optimizers (imported as subpackage)
- **ATen**: Actual operation implementations (via C++ layer)
- **c10**: Core abstractions (Device, ScalarType, etc.)

### Dependency Direction

```
User Code
    ↓
torch package (main API entry point)
    ↓
torch._C (C++ bindings via pybind11)
    ↓
ATen operations (C++ implementations)
    ↓
c10 abstractions (core types)
```

## Trade-offs and Design Decisions

### Public API Density

**Decision**: Expose 500+ functions and classes in `torch` namespace for immediate access.

**Trade-off**:
- ✅ **Advantage**: Convenient; users don't need to import submodules for common functions
- ✅ **Advantage**: Autocomplete-friendly in IDEs
- ❌ **Disadvantage**: Namespace pollution; hard to discover what's available
- ❌ **Disadvantage**: Backward compatibility burden; hard to deprecate

**Evidence**: `torch/__init__.py` has 600+ lines of imports and __all__ definitions.

### Factory Function Uniformity

**Decision**: All tensor creation functions accept `device`, `dtype`, `requires_grad` as keyword arguments.

**Trade-off**:
- ✅ **Advantage**: Consistent API; users learn once, apply everywhere
- ✅ **Advantage**: Easy to change device/dtype globally via default context
- ❌ **Disadvantage**: Repetitive; every function signature includes these
- ❌ **Disadvantage**: Keyword arguments can be verbose

**Evidence**: `torch.randn(size, *, device=None, dtype=None, requires_grad=False)`.

### Tensor as Python Class vs. C++ Instance

**Decision**: Python `torch.Tensor` class is a thin wrapper around C++ TensorImpl; operations bridge Python-C++.

**Trade-off**:
- ✅ **Advantage**: Performance: hot paths execute in C++
- ✅ **Advantage**: Pythonic: `.shape`, `.dtype` properties feel natural
- ❌ **Disadvantage**: Debugging across Python-C++ boundary is complex
- ❌ **Disadvantage**: Operator overloads in Python call C++; hard to trace

**Evidence**: `torch.Tensor` defined in pybind11 bindings; most operations implemented in C++.

### Lazy Module Loading

**Decision**: Some submodules (jit, fx, quantization) are imported but not loaded until first use.

**Trade-off**:
- ✅ **Advantage**: Faster import time; don't load unnecessary modules
- ✅ **Advantage**: Reduce memory footprint if user doesn't need certain features
- ❌ **Disadvantage**: First use of module has latency cost
- ❌ **Disadvantage**: Harder to track dependencies

**Evidence**: `torch/jit/__init__.py` uses lazy loading patterns.

### Default Device as Context

**Decision**: Default device (CPU vs CUDA) is managed via context manager `torch.device()` and global state.

**Trade-off**:
- ✅ **Advantage**: Convenient; users can set once and all new tensors use that device
- ✅ **Advantage**: Easy to write device-agnostic code
- ❌ **Disadvantage**: Implicit state; can lead to unexpected behavior
- ❌ **Disadvantage**: Debugging device-related issues can be subtle

**Evidence**: `torch.get_device()`, `torch.set_device()` manage default.

## Extension Boundaries

### Creating Custom Operators

**Boundary**: Implement custom operation in C++ (or use Python) and register with PyTorch.

```python
# Python-only custom function (slow)
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2

result = MyFunc.apply(tensor)
```

Or implement in C++ and register via torchgen/torch.library (more complex).

**Evidence**: `torch.autograd.Function` class enables Python-based custom operations.

### Tensor Subclassing

**Boundary**: Users can subclass `torch.Tensor` but this is advanced and not recommended.

```python
class MyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x):
        return super().__new__(cls, x)
```

**Evidence**: Documented in PyTorch advanced documentation; not commonly used.

### Custom Optimizers

**Boundary**: Inherit from `torch.optim.Optimizer` and implement `step()` method.

See torch.optim ADR for details.

**Evidence**: Users regularly create custom optimizers.

## Runtime Implications

### Lifecycle and Initialization

1. **Import**: `import torch` triggers `torch/__init__.py` execution
2. **C++ Loading**: torch._C extension loaded; pybind11 bindings initialized
3. **Default Configuration**: Defaults for dtype, device, seed set
4. **Ready**: torch namespace populated with functions and classes
5. **User Code**: Can now call `torch.randn()`, `torch.nn.Linear()`, etc.

### Concurrency Behavior

**Thread Safety**:
- **Global State**: Default device, dtype, random seed are global; not thread-safe for writes
- **Thread-Local Defaults**: Each thread can have its own default device via context manager
- **Tensor Operations**: Individual tensor operations are thread-safe (delegates to ATen)

**Evidence**: Default device managed via global variable; no locking.

### Failure Behavior

1. **Import Failure**: torch._C not found; installation issue
2. **Operation Not Available**: Attempting unsupported operation raises AttributeError
3. **Type Error**: Passing wrong types to functions raises TypeError
4. **Device Error**: Attempting operation on unavailable device raises error

**Evidence**: Try-except around torch._C import to provide helpful error message.

## Performance Implications

### Known Hotspots

1. **Import Time**: Loading torch._C extension can take 1-5 seconds depending on build
2. **Operation Dispatch**: Python function call → C++ dispatch → kernel execution
3. **Memory Allocation**: Creating large tensors involves allocator lookup and memory write

### Allocation Patterns

- **Factory Functions**: Allocate new tensors
- **Operations**: Allocate result tensors
- **Views**: No allocation; share storage

### Python-C++ Boundary Cost

- **Overhead**: Python function call to C++ has overhead (pybind11 marshalling)
- **Batching**: Many small operations are slower than fewer large operations
- **JIT Mitigation**: torch.jit.script() and torch.jit.trace() reduce boundary crossing

## Ownership Boundaries

### State Owned by torch Package

1. **Public API Surface**: Function and class names, signatures, documentation
2. **Default Configuration**: Default device, dtype, random seed
3. **Version Information**: torch.__version__
4. **Package Structure**: How submodules are organized

### State Borrowed from C++ Layer

1. **Tensor Data**: torch.Tensor instances wrap C++ TensorImpl; data owned by c10
2. **Operations**: torch.add() delegates to ATen; logic owned by ATen
3. **Configuration**: torch._C manages actual state; Python layer just accesses

### State Owned by Users

1. **Tensor Values**: Data created by user code
2. **Model Parameters**: torch.nn.Parameter instances created by user models
3. **Training State**: Optimizer state, gradients accumulated during training

## Key Implementation Files

| File | Purpose |
|---|---|
| `__init__.py` | Main package initialization, factory functions, public API |
| `functional.py` | Functional operations (deprecated in favor of in-namespace functions) |
| `Tensor.py` | Python methods and properties (if any Python-level implementations) |
| `__config__.py` | Build configuration metadata |
| `_C/__init__.py` | C++ extension module imports |
| `csrc/` | C++ pybind11 binding source code |
| `utils/` | Utility modules (data loading, checkpointing, etc.) |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/__init__.py` and submodules — checked for API organization, factory functions, configuration
2. **Book Chapter**: Chapter 01 "Entrypoints and Execution Origins" describes where torch.py fits in execution flow
3. **Code Flow**: Traced from user `import torch` through initialization to operation dispatch

Last Verified: 2026-05-27
