# Architecture Decision Record: PyTorch Main Module (torch)

## Architectural Role

**torch** is PyTorch's main Python module and the primary user-facing API surface. It orchestrates initialization of all subsystems (autograd, nn, backends, CUDA) and exposes tensor creation functions, operations, and type objects. When a user writes `import torch`, the torch module's `__init__.py` orchestrates loading of the C extension (`torch._C`), conditional platform-specific library loading, and lazy initialization of subsystems.

**Location**: `torch/` | **Language**: Python + C++ bindings (torch/csrc/) | **Dependencies**: c10, ATen, autograd

## Responsibilities

**torch owns**:
- Python entrypoint and initialization orchestration (`torch/__init__.py`)
- Platform-specific library loading (Windows DLL handling, RTLD_GLOBAL flags on Unix)
- C extension import sentinel checks
- Tensor creation API (`torch.tensor()`, `torch.zeros()`, `torch.ones()`, etc.)
- Tensor operations exposed as free functions (`torch.add()`, `torch.matmul()`, etc.)
- Global configuration (logging, thread pool settings, error handlers)
- Submodule exports and lazy imports (`torch.nn`, `torch.autograd`, `torch.cuda`, etc.)
- Type objects (Tensor, Generator, Device, Stream, ScalarType, etc.)

**torch does not own**:
- Autograd graph construction (torch.autograd owns that)
- Neural network module system (torch.nn owns that)
- Backend drivers (backend modules own those)
- Actual kernel implementations (ATen owns those)

## Dependencies

### Inbound Dependencies
- **Users** import torch and expect tensor creation, operations, and module access
- **torch.nn** imports torch for Tensor type and operations
- **torch.autograd** registers with torch's C extension initialization
- **torch.cuda** imports torch to access Device management
- **torch.optim** imports torch for parameter access

### Outbound Dependencies
- **torch._C** (C extension) provides Tensor type, operations, and backend support
- **torch.csrc** (C++ source) provides C extension implementation
- **ATen** (through torch._C) provides actual tensor operations
- **c10** (through ATen) provides foundational tensor types

## Trade-offs and Design Decisions

### 1. Platform-Specific DLL/Library Loading (Windows vs Unix)
**Decision**: Windows requires manual DLL loading via kernel32 functions; Unix uses RTLD_GLOBAL flags to control symbol visibility.

**Rationale**: 
- Windows lacks LD_LIBRARY_PATH mechanism; must replicate it at application level
- Unix systems use RTLD_GLOBAL to make libtorch symbols visible to subsequently loaded plugins
- Different mechanisms necessary due to OS-level differences in dynamic linking

**Evidence**: `torch/__init__.py:169-289` (Windows) vs lines 425-450 (Unix)

**Trade-off**: Significantly more complex Windows initialization path; adds build-time conditional code.

### 2. C Extension Import with RTLD_GLOBAL
**Decision**: Temporarily modify `sys.getdlopenflags()` before importing torch._C to enable RTLD_GLOBAL, then restore flags afterward.

**Rationale**: Allows multiple PyTorch extensions (custom operators, backend plugins) to share symbols from libtorch without symbol duplication or collisions.

**Trade-off**: Brief global modification of dynamic linker state; minor startup cost.

### 3. Initialization Sentinel Check
**Decision**: After importing torch._C, verify successful import by attempting to access sentinel symbol `torch._C._initExtension`.

**Rationale**: Catches common installation errors (pip install vs pip install -e) that would otherwise cause cryptic downstream failures.

**Evidence**: Lines 1046-1071 in torch/__init__.py

### 4. Conditional ROCm Initialization
**Decision**: Before main C extension loads, check for and call `torch._rocm_init.initialize()` if available.

**Rationale**: ROCm runtime must be prepared before ATen/CUDA subsystems initialize, similar to CUDA initialization sequences.

**Trade-off**: Adds platform-specific branching; only active on ROCm builds.

### 5. Lazy Submodule Imports
**Decision**: Submodules (torch.nn, torch.autograd, torch.cuda) imported lazily or selectively rather than all at import time.

**Rationale**: Reduces initial import overhead; users may only need a subset of functionality. Enables optional backends (CUDA) to not cause import failure if unavailable.

**Trade-off**: First access to lazy module is slower; adds complexity to import system.

### 6. Dual API for Operations: Method vs Free Function
**Decision**: Every operation available as both method (e.g., `x.add(y)`) and free function (e.g., `torch.add(x, y)`).

**Rationale**: 
- Pythonic method form preferred by many users
- Functional form preferred for composition, piping, and explicit argument order
- Both call same underlying implementation

**Implementation**: Generated from ATen via code generation, not hand-written.

## Extension Boundaries

**Custom operators**: Register via `torch.library` or `torch._library` modules; integrated into dispatcher at runtime.

**Backend support**: Implement backend-specific modules (e.g., torch.cuda, torch.xpu) that register device support, memory allocators, and kernels.

**Plugin extensions**: Can register custom dispatch keys with ATen dispatcher (via torch._library) to extend all operations.

## Runtime Implications

### Initialization Sequence
1. Python loads torch/__init__.py
2. Platform-specific DLL loading (Windows) or RTLD_GLOBAL setup (Unix)
3. Optional ROCm initialization
4. Import torch._C (C extension), triggering torch/csrc/Module.cpp's initModule()
5. Verify initialization via sentinel check
6. Set up Python-level API, register hooks
7. Lazy import of submodules

**Total startup overhead**: 10-50ms on typical systems (depends on CUDA availability, platform)

### Lifecycle
- Module-level initialization happens once at first import
- Lazy submodule imports happen on first access
- Backends registered at import time (if available) or first use

### Concurrency
- Initialization is **not thread-safe** for concurrent imports; Python's import lock prevents this
- After initialization, module state is immutable
- Operations are thread-safe at the ATen level (individual tensors are not, but independent tensors can be used concurrently)

### Configuration
- Global settings via `torch.set_num_threads()`, `torch.set_float32_matmul_precision()`, etc.
- Per-operation overrides via context managers (`torch.no_grad()`, `torch.enable_grad()`)

## Performance Implications

### Known Hotspots
1. **Initial import time**: DLL loading (Windows, ~50ms), C extension initialization (~20ms), ROCm initialization (if present, ~100ms)
2. **Lazy import overhead**: First access to torch.nn, torch.cuda, etc. incurs module import cost

### Optimization Opportunities
1. **Lazy loading**: Users who don't use CUDA can avoid CUDA initialization cost
2. **Module precompilation**: PyTorch wheels include precompiled modules to reduce startup
3. **DLL caching**: Windows DLL loading is repeated but quick after first load

### Synchronization Costs
- Minimal synchronization during initialization; mostly sequential
- Backend initialization may include synchronization if initializing GPU support

## Ownership Boundaries

**torch owns**:
- Python-level API surface and initialization
- Tensor creation functions
- Free-function API for operations
- Submodule registry and lazy loading mechanism

**torch delegates to torch._C**:
- Type objects (Tensor, Device, etc.)
- Actual operation implementations (via ATen)
- Backend support (CUDA, HIP, Metal, etc.)

**torch references but does not own**:
- torch.nn (owns neural network module system)
- torch.autograd (owns autograd graph construction)
- torch.optim (owns optimization algorithms)
- torch.distributed (owns distributed training coordination)

**Parent/peer modules own**:
- torch/csrc/ C++ code implements torch._C
- torch/backends/ implement backend-specific functionality
- torch/utils/ implement utility functions
