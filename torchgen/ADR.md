# ADR: torchgen — Code Generation Tools

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

torchgen is the code generation framework that auto-generates PyTorch's operator bindings, dispatch registrations, and C++ type signatures from a single source of truth: `native_functions.yaml`. It is a **build-time tool** that produces source files consumed by ATen and torch/csrc.

torchgen is classified as **Lifecycle Root** (build-time) because it defines the operator surface area that is then compiled into all of PyTorch.

## Responsibilities

### What torchgen Owns
- **Operator definition parsing**: Read native_functions.yaml (16,222 lines)
- **C++ code generation**:
  - `RegisterSchema.cpp` — Operator schema registration (TORCH_LIBRARY blocks)
  - `RegisterDispatchKey.cpp` — Per-backend kernel registration (TORCH_LIBRARY_IMPL blocks)
  - `Functions.h` — C++ operator API surface (auto-generated function stubs)
  - `NativeFunctions.h` — Operator interfaces for kernel implementations
- **Python code generation**:
  - pybind11 wrapper stubs for Python binding
  - Type hints and docstrings
- **Selective build support**: Generate minimal binaries with only requested operators
- **Backend code generation**: Custom code for lazy evaluation, backend-specific optimizations

### What torchgen Does Not Own
- **Kernel implementations**: Implemented manually in aten/src/ATen/native/*.cpp
- **Python API surface**: Defined in torch/ Python code; uses generated stubs
- **Operator dispatch**: Implemented in aten/src/ATen/core/dispatch/Dispatcher
- **Type definitions**: c10 owns core types (Tensor, Scalar, Device, dtype)

## Dependencies

### Build-Time Dependencies (torchgen inputs)
- **native_functions.yaml**: Authoritative definition of all operators
- **parsed_yaml.py**: YAML parsing and validation
- **API types** (api/types/): Type definitions for argument/return types

### Build-Time Dependencies (torchgen outputs consumed by)
- **aten/CMakeLists.txt**: Invokes torchgen to generate ATen code
- **torch/csrc/**: Uses generated pybind11 stubs for Python bindings
- **ATen backends**: Use generated NativeFunctions.h to implement kernels

### Runtime Dependencies (none)
torchgen runs at build time only; no runtime dependencies on generated code (code depends on runtime libraries)

## Trade-offs and Design Decisions

### 1. Single Source of Truth (native_functions.yaml)
**Decision**: All operator definitions centralized in YAML; code generated instead of hand-written  
**Rationale**:
- **Consistency**: Operator signature, Python binding, C++ stub all match
- **Maintainability**: Adding new operator requires only YAML entry; code generation produces bindings
- **Type safety**: Generated code is type-checked against YAML schema

**Trade-off**: YAML parsing errors are opaque; build failures delay discovery until code generation

**Evidence**: `native_functions.yaml` is 16,222 lines. `torchgen/gen.py` parses and processes entire file. Changes to operator signatures trigger full regeneration of RegisterSchema.cpp, RegisterDispatchKey.cpp, Functions.h.

### 2. Separation of Schema and Implementation Registration
**Decision**: TORCH_LIBRARY and TORCH_LIBRARY_IMPL generated separately, enabling per-backend customization  
**Rationale**:
- **Operator reuse**: One schema + multiple implementations (CPU, CUDA, Sparse, Quantized)
- **Lazy backend loading**: Schemas always present; implementations loaded on first use
- **Type validation**: Dispatcher validates kernel signature against schema at registration

**Trade-off**: Two-phase registration adds build complexity

**Evidence**: `torchgen/dest/register_schema.py` generates schema registrations. `torchgen/dest/register_dispatch_key.py` generates per-backend registrations.

### 3. Code Generation Over Manual Stubs
**Decision**: Generate boilerplate instead of maintaining by hand  
**Rationale**:
- **Automation**: 2,500 operators → 2,500 schemas, stubs, registrations auto-generated
- **Consistency**: Generated code follows uniform patterns; no drift
- **Refactoring**: Changes to operator structure propagate automatically

**Trade-off**: Generated code is verbose and hard to read; compilation times increase due to generated file volume

**Evidence**: `RegisterSchema.cpp` generated from native_functions.yaml is ~100KB. `Functions.h` is similarly large.

### 4. Selective Build Support
**Decision**: Generate optimized operator sets for embedded/mobile builds  
**Rationale**:
- **Size reduction**: Exclude unused operators from binary (saves 10-100MB for mobile)
- **Build time**: Smaller operator sets reduce compilation time
- **Customization**: Each embedded platform can build only needed operators

**Trade-off**: Requires curating operator lists; fragile if dependencies missed

**Evidence**: `torchgen/selective_build/` contains selective build logic. `gen_backend_stubs.py` generates operator headers filtered by operator lists.

## Extension Boundaries

### Public Extension Points
1. **Adding new operators**: Update native_functions.yaml, run torchgen, implement kernels in aten/src/ATen/native/
2. **Backend customization**: Custom TORCH_LIBRARY_IMPL registrations for new device types
3. **Operator decomposition**: Define composite operators in terms of other operators (torchgen generates registrations)

### Extension Constraints
- **native_functions.yaml schema is fixed**: Operators must conform to YAML structure (func, variants, manual_cpp_binding, etc.)
- **Operator names are global**: Cannot redefine operators from built-in set
- **Type system is closed**: Only types defined in api/types/ are supported; new types require torchgen changes

## Runtime Implications

### Build-Time Processing
1. **Parsing**: torchgen reads native_functions.yaml (milliseconds)
2. **AST construction**: Build in-memory representations of all operators (milliseconds)
3. **Code generation**: Generate RegisterSchema.cpp, RegisterDispatchKey.cpp, Functions.h (seconds for 2,500 operators)
4. **Output files written**: Generated files committed to build output directory (milliseconds)
5. **Compilation**: C++ compiler reads generated files and compiles into objects (minutes for full ATen)

### Generated Code Characteristics
- **RegisterSchema.cpp**: ~100-200KB, ~5,000 lines (2,500 operators × 2 lines/operator)
- **RegisterDispatchKey.cpp**: Similar size, per-backend (CPU, CUDA, Sparse, Quantized, etc.)
- **Functions.h**: ~500-1000KB, huge header with all operator signatures

### Incremental Build Behavior
- **Full rebuild** if native_functions.yaml changes (regenerates all files)
- **Partial rebuild** if single kernel implementation changes (only that backend's objects recompile)
- **No regeneration** if torchgen changes (unless invoked manually)

## Performance Implications

### Build-Time Overhead
- **Torchgen execution**: ~5-10 seconds for full code generation (single-threaded Python script)
- **Generated file compilation**: ~5-10 minutes for full ATen library (parallelized with -j flags)
- **Incremental build**: Milliseconds if no YAML changes; seconds if YAML changed

### Runtime Characteristics
- **Generated code is zero-overhead**: Compiles to identical binary to hand-written code
- **Inlining opportunities**: Inline function stubs generated by template; optimizer can eliminate wrappers
- **Code size**: Generated code is verbose but compresses well (large binaries, but mostly debug symbols)

## Ownership Boundaries

### Code Generation Responsibilities
- **native_functions.yaml** — Owned collaboratively; changes coordinated across team
- **Operator definitions** — Added by feature authors; torchgen consumes and generates code
- **Generated output files** — Owned by torchgen (not committed to repo; generated at build time)
- **Backend registrations** — Authors of backend implementations provide TORCH_LIBRARY_IMPL blocks

## Notes and Caveats

1. **native_functions.yaml is the single source of truth**: Changes to operator signatures must go through YAML; hand-editing generated files is not supported
2. **Code generation is deterministic**: Same YAML → same generated code; no randomness or non-deterministic side effects
3. **Generated code is not committed**: Temporary output; build system always regenerates
4. **Large operator definitions can cause build slowdowns**: Each operator definition adds lines to generated files; thousands of operators → verbose code → longer compilation
5. **Operator name changes are risky**: Renaming operators in YAML breaks old code; no automated migration tool
6. **Selective build is fragile**: Missing dependencies in operator lists causes runtime "operator not found" errors; requires careful curation
7. **Type system is closed**: Adding new operator argument types requires modifying torchgen's type definitions and regenerating everything
