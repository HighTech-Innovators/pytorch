# Architecture Decision Record: ATen Tensor Operations Library

## Architectural Role

**ATen** (A Tensor Library) is PyTorch's tensor operations engine — the layer that defines, implements, and dispatches all tensor operations. While c10 provides the core tensor *type* (`TensorImpl`), ATen provides the *operations* that act on tensors (`add`, `mul`, `matmul`, `conv2d`, etc.). It is the central hub where tensor operations are defined, routed to appropriate backends, and executed.

**Location**: `aten/src/ATen/` | **Language**: C++ with code generation | **Dependencies**: c10

## Responsibilities

**ATen owns**:
- Operator definition and registry (2,691 operators defined in `native_functions.yaml`)
- Central dispatcher for routing operations to kernels based on DispatchKeySet
- Tensor wrapper API (`at::Tensor`, `at::TensorBase`)
- Kernel implementations for supported backends (CPU, CUDA, HIP, Metal, MPS)
- CPU instruction-set dispatch (AVX2, AVX512) via `DispatchStub.h`
- Operator metadata and type boxing infrastructure
- Native operation definitions (elementwise, linear algebra, convolution, etc.)

**ATen does not own**:
- Autograd graph construction (autograd engine owns that, but wraps ATen kernels)
- Python bindings (torch/csrc owns that, but wraps ATen operators)
- c10 tensor types (c10 owns those)
- Backend hardware initialization and management (backends own that)

## Dependencies

### Inbound Dependencies
- **autograd** wraps ATen operations to track gradients
- **torch.nn** composes ATen operations into higher-level layers
- **Python** imports ATen via torch._C bindings (torch/csrc)
- **All backend implementations** register kernels with ATen's dispatcher

### Outbound Dependencies
- **c10** for tensor types (TensorImpl, StorageImpl, Device, DispatchKey)
- **torchgen** (code generation) produces most of ATen's API from YAML definitions

## Trade-offs and Design Decisions

### 1. Code Generation from YAML (native_functions.yaml)
**Decision**: Define 2,691 operators in `native_functions.yaml`; generate C++ wrappers, dispatcher registrations, and Python bindings.

**Rationale**: Avoids massive duplication. A single operator definition produces:
- C++ method (`Tensor::add()`)
- C++ free function (`torch::add()`)
- Dispatcher registration entry
- Python method wrapper
- Generated documentation

**Alternative**: Hand-write every variant for every operator (unmaintainable)

**Trade-off**: Build-time code generation is complex; harder to debug generated code.

### 2. Dispatcher: Central Router via DispatchKeySet
**Decision**: Operations route through a central `Dispatcher` singleton that checks `DispatchKeySet` to select the appropriate kernel implementation.

**Rationale**: Enables:
- Multi-backend support (CPU, CUDA, Metal, etc.) without per-operation branching
- Composable features: autograd, tracing, profiling, custom kernels registered as dispatch keys
- Clean separation between operator definition and implementation

**Evidence**: Source shows `Dispatcher` in `aten/src/ATen/core/dispatch/Dispatcher.h` with `OperatorEntry` registry.

### 3. Tensor: Lightweight Wrapper Around TensorImpl
**Decision**: `at::Tensor` is a thin wrapper (`intrusive_ptr<TensorImpl>`) with generated methods; not a base class for kernels.

**Rationale**: Kernels work with `TensorImpl` directly (or `TensorBase` for read-only access) to avoid method call overhead and allow in-place mutations.

**Trade-off**: More API surfaces (Tensor methods, free functions, dispatcher entries) for same operation.

### 4. Three Operation Variants: out-of-place, in-place, out
**Decision**: For operators like `add`, provide:
- `t1.add(t2)` — out-of-place, returns new tensor
- `t1.add_(t2)` — in-place, mutates t1 and returns it
- `torch::add(t1, t2, out=t3)` — out variant, fills pre-allocated t3

**Rationale**: 
- Out-of-place is safe and idiomatic in Python
- In-place enables memory efficiency in loops
- Out variant is useful for pre-allocated GPU memory scenarios

**Implementation**: Generated from single `native_functions.yaml` entry with variant specifications.

### 5. CPU Instruction Set Dispatch (DispatchStub)
**Decision**: CPU kernels use `DispatchStub.h` to dispatch at runtime based on available CPU instruction sets (AVX2, AVX512, etc.).

**Rationale**: Single binary works across different CPU architectures; performance-critical kernels benefit from vectorization when available.

**Evidence**: BinaryOps.cpp and other CPU implementations use `DispatchStub` for kernel selection.

### 6. Separation of Backend-Specific Code
**Decision**: Kernel implementations organized by backend: `native/cpu/`, `native/cuda/`, `native/hip/`, etc.

**Rationale**: Backend-specific kernels (e.g., CUDA) are only compiled when backend is enabled. CPU kernels always compiled.

**Trade-off**: Duplicated operation structure across backends, but enables backend-specific optimizations and conditional compilation.

## Extension Boundaries

**New operators**: Define in `native_functions.yaml`; code generation produces wrappers and dispatcher entries. Implement kernels in `native/` subdirectories for each backend.

**New backends**: Implement `Allocator` interface (c10 responsibility) and register kernels for all operators via dispatcher. Build system conditionally compiles backend-specific code.

**Custom kernels**: Register with dispatcher at runtime via `Dispatcher::registerKernel()`. Can override built-in kernels or add new dispatch keys.

**Dispatch keys**: New behaviors (e.g., new backend, profiling, custom autograd) add new DispatchKey; register kernels for existing operators under that key.

## Runtime Implications

### Initialization
- `native_functions.yaml` is processed at build time; no runtime cost
- Dispatcher initializes with pre-registered operator entries
- Backend allocators registered at first use (lazy initialization for optional backends like CUDA)

### Kernel Dispatch Execution
1. User calls `tensor.add(other)`
2. Calls dispatcher with `DispatchKeySet` from both operands
3. Dispatcher selects kernel based on DispatchKeySet (e.g., CPU, AutogradCPU if requires_grad)
4. Kernel executes (e.g., CPU vectorized operation or CUDA kernel)
5. Returns new Tensor wrapping result TensorImpl

**Dispatch overhead**: Very small (bitset lookup, hash table lookup); typically < 1% of kernel execution time.

### Memory and Allocation
- Operations allocate result tensors via c10 memory management
- In-place operations reuse storage; may allocate if broadcast semantics require it
- "Out" variant uses provided storage; fails if sizes don't match

### Concurrency
- Individual tensor operations are **not thread-safe** for concurrent mutations
- Immutable tensor data can be safely read from multiple threads
- Backends (CUDA, HIP) handle device synchronization; CPU operations are synchronous

### Lifecycle
- Dispatcher lives for process duration
- Kernel registrations are permanent; no unregistration
- Operator definitions are immutable after registration

## Performance Implications

### Known Hotspots
1. **Dispatch overhead**: Bitset membership test, hash table lookup for operator entry, kernel function pointer call
2. **Memory allocation**: Every out-of-place operation allocates new result tensor
3. **Data movement**: CPU↔GPU transfers when operands on different devices (explicit, slow)
4. **Kernel execution time**: Dominates for most operations; dispatch overhead negligible by comparison

### Optimization Opportunities
1. **Fusion**: Multiple operations fused into single kernel (handled by higher layers like inductor)
2. **In-place operations**: Reuse storage, avoid allocations
3. **CPU vectorization**: AVX2/AVX512 for elementwise operations
4. **Backend-specific kernels**: CUDA kernels heavily optimized for GPU memory hierarchy

### Synchronization Costs
- No explicit synchronization in ATen; backends handle it
- CUDA operations are asynchronous; Python bindings insert synchronization points where needed

## Ownership Boundaries

**ATen owns**:
- Operation semantics (what add/mul/matmul mean)
- Dispatcher and kernel routing
- Tensor wrapper API
- Kernel implementations (at least the CPU reference implementations)

**ATen borrows**:
- TensorImpl from c10
- Backend-specific optimizations (CUDA kernels often provided by hardware vendors or highly optimized in-house)

**Parent layers own**:
- Autograd graph (autograd engine wraps ATen operations to track gradients)
- Python bindings (torch/csrc wraps ATen operations)
- High-level module API (torch.nn composes ATen operations)

**Backend providers own**:
- CUDA kernel implementations (NVIDIA)
- ROCm/HIP implementations (AMD)
- Metal implementations (Apple)
