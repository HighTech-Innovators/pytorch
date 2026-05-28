# ADR: torch Library — Python API and Core Components

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The torch library is PyTorch's user-facing Python API and the bridge between users and the underlying C++ execution engine. It encompasses:

1. **torch.core (torch/csrc)**: Python/C++ boundary via pybind11 and CPython C API
2. **torch.autograd**: Automatic differentiation engine for gradient computation
3. **torch.nn**: Neural network building blocks (layers, containers)
4. **torch.jit**: TorchScript compiler for graph optimization and deployment
5. **torch.distributed**: Multi-process training coordination

The torch library is classified as **Runtime Critical** because it serves as the primary user-facing interface and owns critical execution components (autograd engine).

## Responsibilities

### torch.core (torch/csrc) — Python/C++ Bridge
- **Module initialization**: Initialize torch._C C extension at Python import time
- **Type registration**: Register Tensor, Storage, Device, dtype types for Python
- **Function bindings**: pybind11 wrappers for ATen operators, autograd functions
- **Argument marshaling**: Convert Python objects to C++ types and back
- **Exception handling**: Translate C++ exceptions to Python exceptions
- **GIL management**: Acquire/release Python GIL for thread safety

### torch.autograd — Automatic Differentiation
- **Graph construction**: Record forward operations in a computation graph (DAG)
- **Backward computation**: Execute backward pass to compute gradients
- **Gradient accumulation**: Sum gradients at leaf tensors
- **Custom autograd**: User-defined differentiation via torch.autograd.Function
- **Mode control**: Enable/disable autograd (requires_grad, no_grad, inference_mode)
- **Anomaly detection**: Detect numerical issues in backward pass

### torch.nn — Neural Network Abstractions
- **Modules**: Base class for all neural network components
- **Parameters**: Learnable tensors (weights, biases) tied to modules
- **Layers**: Standard building blocks (Linear, Conv2d, RNN, etc.)
- **Containers**: Structures for organizing modules (Sequential, ModuleList, ModuleDict)
- **Parameter management**: Serialization, gradient access, initialization
- **Hooks**: Forward/backward hooks for monitoring and modifying computation

### torch.jit — TorchScript Compiler
- **Graph IR**: Intermediate representation for traced/scripted code
- **Tracing**: Capture computational graph from eager execution
- **Scripting**: Compile Python-like code to graphs
- **Optimization**: Fuse operations, eliminate dead code, specialize types
- **Serialization**: Save/load compiled graphs as TorchScript
- **Deployment**: Execute graphs without Python interpreter

### torch.distributed — Distributed Training
- **Process groups**: Synchronization across multiple processes
- **Collective operations**: AllReduce, AllGather, Broadcast for gradient synchronization
- **Distributed data parallel**: Data parallelism with automatic gradient synchronization
- **Backend abstraction**: Support multiple communication backends (NCCL, Gloo, MPI)
- **Process launching**: Initialization and coordination of worker processes

### What torch Does Not Own
- **Kernel implementations**: Owned by aten/src/ATen/native/
- **Tensor representation**: Owned by c10 (TensorImpl, Storage, Device)
- **Dispatch system**: Owned by ATen dispatcher (aten/src/ATen/core/dispatch/)
- **Memory allocation**: Owned by c10 allocators (c10/core/Allocator*)
- **Low-level utilities**: Owned by c10 (intrusive_ptr, ArrayRef, logging)

## Dependencies

### Internal Dependencies (torch → other modules)
- **ATen**: Every operator call goes through ATen dispatcher
- **c10**: TensorImpl, Device, DispatchKey, memory allocators, utilities
- **caffe2** (optional): Legacy code, some functions imported
- **External libraries** (optional): pybind11, NCCL, Gloo, oneDNN, XNNPACK

### External Dependencies (other modules → torch)
- No other core PyTorch subsystems depend on torch; torch depends on everything
- torch is the top of the dependency chain (Python-facing)

## Trade-offs and Design Decisions

### 1. Python API (torch) vs C++ Core (c10, ATen)
**Decision**: Expose high-level ergonomic Python API while keeping performance-critical code in C++  
**Rationale**:
- **Pythonic API**: Dynamic typing, operator overloading, syntactic sugar for research
- **C++ performance**: Tensor operations execute natively; interpretation overhead negligible
- **Portability**: C++ code compiles to platforms without Python interpreter (mobile, edge)

**Trade-off**: Maintenance burden of two codebases and pybind11 bindings

**Evidence**: `torch/__init__.py` (~200 lines) imports torch._C and wraps C++ functions. `torch/csrc/Module.cpp` (3,264 lines) registers 47+ entrypoints.

### 2. Module Registry vs Direct Instantiation
**Decision**: torch.nn.Module maintains a global registry of parameters and buffers; submodules discovered via `_modules` dict  
**Rationale**:
- **Parameter management**: Modules can enumerate parameters without explicit tracking
- **Serialization**: state_dict() works for any module structure
- **Initialization**: Parameters initialized uniformly (size inference from usage)

**Trade-off**: Hidden coupling through global registry; requires careful tracking of module membership

**Evidence**: `torch/nn/modules/module.py` defines `_modules` dict (line ~50). `register_parameter()` and `register_buffer()` update registry.

### 3. Eager Execution vs Graph Compilation
**Decision**: torch defaults to eager execution; TorchScript is opt-in for compilation  
**Rationale**:
- **Ease of use**: Python control flow works naturally; no special syntax for conditionals
- **Debuggability**: Step through Python code with pdb; inspect tensors at any point
- **Flexibility**: Tensors shapes can be dynamic; control flow depends on data

**Trade-off**: Slower execution than compiled graphs; operator dispatch overhead accumulates

**Evidence**: Default behavior is eager. `@torch.jit.script` decorator opt-in.

### 4. Autograd Graph as Python Objects
**Decision**: Backward graph (Nodes, Edges) live in C++ but are exposed via Python API (grad_fn, backward())  
**Rationale**:
- **Integration**: Users can inspect grad_fn to understand gradient flow
- **Custom autograd**: torch.autograd.Function allows user-defined backward passes
- **Control**: Users explicitly call .backward(); can control gradient propagation

**Trade-off**: Circular references between Python tensors and C++ graph nodes require careful garbage collection

**Evidence**: `torch/csrc/autograd/python_variable.cpp` exposes grad_fn property. `torch.autograd.Function` allows custom backward implementation.

### 5. Distributed Data Parallel via Gradient Hooks
**Decision**: Use autograd hooks to synchronize gradients across processes after each backward  
**Rationale**:
- **Compatibility**: Works with any optimizer; no changes to training loop needed
- **Transparency**: Gradient synchronization is automatic, users don't explicitly sync
- **Performance**: Overlaps gradient computation and communication (if implemented)

**Trade-off**: Adds overhead to every backward pass (even single-GPU training); latency-sensitive for small batches

**Evidence**: `torch/distributed/nn/parallel/data_parallel.py` registers backward hooks. `torch.nn.parallel.DataParallel.reducer()` allocates.

### 6. JIT Graph Specialization
**Decision**: TorchScript graphs are specialized for concrete tensor shapes and dtypes at first execution  
**Rationale**:
- **Performance**: Specialized graphs enable aggressive optimization (constant folding, shape propagation)
- **Type safety**: Type information allows type-checking at compile time

**Trade-off**: Dynamic shapes require graph re-specialization at each shape change; overhead for shape-dynamic models

**Evidence**: `torch/csrc/jit/runtime/profiling_record.cpp` implements type profiling. `type_inference.cpp` specializes graphs.

## Extension Boundaries

### Public Extension Points
1. **Custom modules**: Subclass torch.nn.Module and override forward()
2. **Custom autograd**: Subclass torch.autograd.Function and implement forward() / backward()
3. **Custom operators**: Register via TORCH_LIBRARY macro (exposed to Python via torch.ops)
4. **Distributed backends**: Register via torch.distributed.Backend (Gloo, NCCL, MPI, etc.)
5. **JIT custom ops**: Register TorchScript operators via TORCH_LIBRARY_IR_FRAGMENT

### Extension Constraints
- **Module forward() signature matters**: Affects how TorchScript traces
- **Autograd backward() signature must match forward()**: Type checking at registration
- **JIT requires explicit type annotations**: Dynamic Python code doesn't JIT well
- **Distributed operations must be deterministic**: Non-determinism breaks gradient synchronization

## Runtime Implications

### Initialization
1. **Python import `torch`**: Loads torch._C extension via initModule()
2. **Operator registration**: All kernels pre-registered by ATen static initializers
3. **Submodule imports**: torch.nn, torch.optim, torch.jit imported on demand
4. **Lazy initialization**: NumThreads, CUDA context initialized on first use

### Execution Flow: Python to C++ to Kernel
1. **Python call** — `torch.add(a, b)` or `a + b`
2. **Python binding** — torch/csrc/ pybind11 wrapper calls ATen dispatcher
3. **ATen dispatch** — Reads tensor dispatch keys, selects kernel
4. **Kernel execution** — C++ or CUDA code computes result
5. **Autograd wrapping** — If requires_grad=True, wrap result with grad_fn
6. **Return to Python** — Result converted back to Python Tensor object

### Backward Pass Execution
1. **loss.backward()** — Python calls loss tensor's backward() method (torch/csrc/autograd/python_variable.cpp)
2. **Engine::execute()** — C++ autograd engine initializes backward task graph
3. **Dependency computation** — Traverse graph to compute in-degrees
4. **Ready queue scheduling** — Process nodes in reverse topological order
5. **Gradient accumulation** — Each node sums gradients from downstream and applies its backward function
6. **Leaf tensor gradients** — Accumulated gradients stored in .grad attributes

### Thread Safety
- **Tensor objects are not thread-safe**: Multiple threads must not call .backward() on the same graph
- **GIL management**: Autograd releases GIL during kernel execution, reacquires for Python callbacks
- **Gradient accumulation**: Atomic operations used for in-place accumulation

### Performance Characteristics
- **Python overhead**: ~50-100ns per operator call (pybind11 marshaling)
- **GIL release/reacquire**: ~100-200ns per kernel call
- **Graph construction overhead**: ~100-200ns per operation when requires_grad=True
- **Backward pass overhead**: ~200-400ns per node in computation graph

## Performance Implications

### Memory Overhead
- **Python Tensor wrapper**: ~128 bytes per Python object (plus C++ TensorImpl)
- **Autograd graph**: ~64 bytes per Node + ~32 bytes per Edge
- **Saved variables**: Depends on tensor size (stored full tensor or metadata only)
- **Module parameters**: ~64 bytes per parameter (name + weak_ref + tensor_ptr)

### Performance-Critical Paths
1. **Operator invocation** (every tensor operation)
   - Bottleneck: pybind11 argument marshaling, GIL release/reacquire
   - Optimization: Use C++ API directly when possible; batch operations to amortize overhead
2. **Backward pass** (training)
   - Bottleneck: Graph traversal, gradient accumulation, memory bandwidth
   - Optimization: TorchScript fusion, gradient accumulation in reduced precision
3. **Module parameter access** (initialization, serialization)
   - Bottleneck: Traversing module tree, checking registry
   - Optimization: Cache parameter lists, batch operations

### Synchronization Costs
- **Autograd graph access**: No locks in hot path; operations are single-threaded
- **Distributed gradient synchronization**: Collective operation latency (microseconds to milliseconds depending on cluster size)
- **Module parameter updates**: No synchronization needed; GIL ensures atomicity of Python-level operations

## Ownership Boundaries

### torch/csrc Ownership
- pybind11 bindings and CPython C API wrappers
- Type conversions (Tensor, Storage, Device, dtype, device string parsing)
- Exception handling (TORCH_CHECK → Python exception translation)
- GIL management (releasing before kernel calls, reacquiring for callbacks)

### torch/autograd Ownership
- Graph construction (set_grad_fn, grad_fn_variable)
- Backward execution (Engine, ReadyQueue, GraphTask)
- Saved variable management (SavedVariable, SavedTensor)
- Mode control (requires_grad, no_grad, inference_mode)

### torch/nn Ownership
- Module hierarchy (Module base class, parameter registry)
- Standard layers (Linear, Conv2d, RNN, Transformer)
- Parameter initialization
- Module serialization (state_dict, load_state_dict)

### torch/jit Ownership
- Graph IR (JIT AST, SSA conversion)
- Tracing and scripting (torch.jit.trace, @torch.jit.script)
- Graph optimization passes
- TorchScript serialization

### torch/distributed Ownership
- Process group initialization (init_process_group, get_rank, get_world_size)
- Collective operations (all_reduce, all_gather, broadcast)
- Distributed data parallel (DistributedDataParallel wrapper)
- Communication backend implementations (NCCL, Gloo, etc.)

## Notes and Caveats

1. **GIL is held during Python-level operations**: Custom C++ code can release GIL if needed, but must reacquire before calling Python callbacks
2. **Circular references between Tensor and grad_fn**: Requires Python GC to break cycles; can lead to reference leaks if not careful
3. **Gradient accumulation is in-place**: Multiple .backward() calls on the same graph accumulate gradients in .grad; must call .zero_grad() between passes
4. **DataParallel synchronization is synchronous**: Blocks after each backward until all processes complete gradient sync
5. **JIT graphs are specialized per tensor shape**: Reshaping tensors triggers re-specialization; significant overhead for dynamic shapes
6. **torch.distributed requires external initialization**: Must call dist.init_process_group() before using collective operations; common source of hangs
7. **Autograd graph is not serializable**: TorchScript can serialize graphs, but Python-level grad_fn links are lost; requires state_dict + frozen modules
