# Architecture Decision Record: Code Generator (torchgen)

## Architectural Role

**torchgen** is PyTorch's code generation system — the build-time tool that generates vast amounts of C++ and Python code from declarative specifications. It ingests `native_functions.yaml` (which defines all 2,691 operators) and generates dispatcher entries, C++ kernel wrappers, Python bindings, and backward functions. This avoids massive code duplication and enables consistent API surface across Python and C++.

**Location**: `torchgen/` | **Language**: Python | **Dependencies**: None (build-time only, no runtime dependencies)

## Responsibilities

**torchgen owns**:
- Parsing of `native_functions.yaml` operator definitions
- C++ code generation: dispatcher entries, kernel signatures, wrappers
- Python binding generation: method wrappers, free functions, module initialization
- Backward function generation (gradient functions for differentiable operations)
- Type system and code template management
- Dispatch key handling (CPU, CUDA, Autograd, custom backends)
- Functionalization and view handling
- AOTI (ahead-of-time-interpreted) code generation for mobile
- Code generation for lazy tensor backend

**torchgen does not own**:
- Actual kernel implementations (ATen/backends own those)
- Runtime dispatch (dispatcher owns that)
- Autograd computation (autograd engine owns that)
- Python AST or general-purpose code generation (standard libraries own that)

## Dependencies

### Inbound Dependencies
- **Build system** invokes torchgen to generate code from `native_functions.yaml`
- **Developers** modify `native_functions.yaml` to add/change operations
- **CI/build process** runs torchgen as build step

### Outbound Dependencies
- **YAML library** to parse native_functions.yaml
- **Jinja2-like templates** for code generation
- **Standard Python libraries** (ast, pathlib, collections, etc.)

## Trade-offs and Design Decisions

### 1. Single Source of Truth: native_functions.yaml
**Decision**: All operator definitions in one YAML file; code generated from this single source.

**Rationale**: 
- Single source of truth prevents drift between Python and C++ APIs
- Adding operator requires only YAML change; all code automatically generated
- Massive duplication reduction: one YAML entry generates ~10 C++/Python files

**Example**:
```yaml
- name: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  cpu: add_cpu_
  cuda: add_cuda_
  autograd: add_backward
```

Generates:
- C++ dispatcher entry
- C++ method wrapper
- C++ free function
- Python method binding
- Python free function binding
- Backward function registration (if autograd entry specified)

**Trade-off**: Initial setup complex; but massive long-term maintenance savings.

### 2. Code Generation via Templates
**Decision**: Use template system (similar to Jinja2) to generate code from patterns.

**Rationale**: 
- Repetitive code (boilerplate, wrappers) generated from templates
- Customizable per operation: templates can branch on operation type
- Avoids hand-written code duplication

**Evidence**: torchgen/templates/ contains code templates.

### 3. Structured Kernels and Dispatch
**Decision**: Kernels can be "structured" (specify backend, signature pattern) or "composite" (composed of other ops).

**Rationale**: 
- Structured kernels: follow standard pattern; codegen handles dispatch
- Composite kernels: defined as composition; automatically decomposed
- Reduces boilerplate for common patterns

**Example**:
```yaml
- name: relu(Tensor self) -> Tensor
  structured: true
  cpu: relu_cpu
  cuda: relu_cuda
```

### 4. Backward Function Generation
**Decision**: Backward functions auto-generated from forward signature + backward spec.

**Rationale**: 
- Forward and backward must be kept in sync; code generation enforces this
- Backward function registered automatically with dispatcher
- SavedVariable management automatic

**Evidence**: torchgen generates backward functions for all differentiable ops.

### 5. Multiple Dispatch Key Support
**Decision**: Single operation specifies implementation for multiple dispatch keys (CPU, CUDA, HIP, custom).

**Rationale**: 
- Operations have different implementations per backend
- YAML specifies which backends have which implementation
- Code generator produces dispatch routing

**Example**:
```yaml
- name: add
  cpu: add_cpu_
  cuda: add_cuda_
  hip: add_hip_
  xpu: add_xpu_
```

### 6. Type Bindings and Python Conversion
**Decision**: Generator handles type conversion between C++ types and Python types.

**Rationale**: 
- Operations defined in C++ with C++ types (at::Tensor, torch::Tensor, int64_t, etc.)
- Python needs different types (torch.Tensor, int, etc.)
- Generator bridges the gap via type mapping

**Evidence**: torchgen/api/types.py defines type system and conversions.

### 7. Build-Time Code Generation
**Decision**: Code generation happens at build time, not runtime.

**Rationale**: 
- Compilation time overhead acceptable (one-time during build)
- Runtime code is hand-optimized/precompiled
- Generated code can be inspected and debugged

**Trade-off**: Build time increases (~1-5 minutes depending on hardware); only happens during build.

### 8. ViewOps and Functionalization Handling
**Decision**: Special handling for view operations and functionalization layer.

**Rationale**: 
- View operations (reshape, transpose) create new metadata, same storage
- Functionalization layer converts in-place ops to functional equivalents
- Special code generation for correct semantics

**Evidence**: gen_functionalization_type.py handles functionalization generation.

## Extension Boundaries

**New operations**: Add entry to `native_functions.yaml` with name, signature, backend implementations, backward spec. Rebuild; code automatically generated.

**New backends**: Add new dispatch key, implement kernels, add to YAML entries. Generator automatically produces dispatch routing.

**Custom code generation**: Extend torchgen with custom generator classes; compose with existing generators.

**Code templates**: Customize code generation by modifying or adding templates in torchgen/templates/.

## Runtime Implications

### Build Process
1. `native_functions.yaml` parsed by torchgen
2. C++ code generated to aten/src/ATen/core/ops/ and related dirs
3. Python binding code generated to torch/_C sources
4. All code compiled into libtorch and torch._C extension
5. Final binary includes pre-compiled generated code

**Build time**: ~5-10% of total PyTorch build time (~20-50 minutes on typical hardware).

### No Runtime Impact
- Generated code is static; no runtime overhead
- All generation happens at build time
- Runtime just executes pre-generated code

### Code Size
- Generated code: ~1-5MB (significant, but negligible vs binary size)
- Reduction in duplication: source code ~20% smaller without generation

### Maintainability
- Source files: Much smaller (yaml only, not full C++/Python)
- Consistency: All apis guaranteed consistent across Python/C++
- Evolution: Changes to native_functions.yaml automatically propagate

## Ownership Boundaries

**torchgen owns**:
- Code generation logic and templates
- Parsing of operator definitions
- Type system for code generation
- Dispatch and backend routing logic

**torchgen uses**:
- native_functions.yaml (users/developers write this)
- YAML parser (standard library)
- Code template engine (custom, written in Python)

**Generated code is owned by**:
- Resulting C++ files (part of ATen)
- Resulting Python binding files (part of torch._C)

**Users/developers own**:
- native_functions.yaml content
- Backend implementations (kernels)
- Templates (if customized)

## Performance Implications

### Build-Time Impact
- Generation adds 5-10% to PyTorch build time
- Incremental builds only regenerate changed sections
- Parallel generation reduces wall-clock time

### Runtime Impact
- **Zero**: Generated code is pre-compiled; identical to hand-written (modulo optimization)

### Generated Code Quality
- Generator optimizes for common patterns
- Hand-written backward functions often more optimized than generated
- Compiler optimizations eliminate differences

## Validation and Testing

**Consistency checks**: torchgen validates that:
- Forward and backward signatures match
- Type compatibility
- Backend specifications complete

**Testing**: Generated code verified by:
- Unit tests on each operation
- Integration tests with autograd
- Numerical gradient checks

