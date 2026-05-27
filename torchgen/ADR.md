# Architecture Decision Record: TorchGen — Operator Code Generation

**Location:** `./src/torchgen/`  
**Last Updated:** 2026-05-27  
**Classification:** Build-Time Tool, Code Generation

---

## Executive Summary

TorchGen is a code generation system that automatically generates C++ stubs and bindings for PyTorch operations. It reads operator declarations from YAML files, parses them, and generates dispatch stubs, backend-specific code, and Python bindings. This eliminates manual boilerplate and ensures consistency across all operation implementations.

---

## Architectural Role

TorchGen serves as the **operator metadata processor and code generator** for PyTorch. Its responsibilities are:

1. **Parse Operator Declarations** — Read YAML files defining operation signatures and semantics
2. **Generate Dispatch Stubs** — Create C++ code that routes operations to correct kernels
3. **Generate Bindings** — Create Python-C++ bindings for each operation
4. **Backend-Specific Code** — Generate device-specific dispatch code
5. **Documentation** — Generate operation documentation from declarations
6. **Validation** — Verify operator declarations are consistent and complete

Because TorchGen generates the operation dispatch layer, **all operations ultimately depend on generated code being correct**.

---

## Responsibilities

### What TorchGen Does

- **YAML Parsing** — Parse operator declarations (native_functions.yaml, etc.)
- **Type Analysis** — Extract argument types and return types
- **Dispatch Code Generation** — Generate routing logic for device/dtype dispatch
- **Binding Generation** — Create pybind11 bindings from declarations
- **Backend Stubs** — Generate backend-specific operation stubs
- **Operation Registration** — Generate TORCH_LIBRARY macro expansions
- **Consistency Checking** — Verify all operations have required backends

### What TorchGen Does NOT Do

- **Implement Kernels** — Code generation produces structure, not algorithm
- **Execute at Runtime** — All code generation happens at build time
- **Modify Declarations** — Works only from provided YAML declarations
- **Handle Edge Cases** — Relies on backend implementations for special cases

---

## Dependencies

### What Depends On TorchGen

- **ATen Build** — ATen generation requires TorchGen output
- **PyTorch Build** — Python bindings built from TorchGen output
- **Operator Registration** — Dispatch table constructed from generated code

### What TorchGen Depends On

- **YAML Declarations** — Native operator declarations (native_functions.yaml)
- **Python** — TorchGen itself is written in Python
- **Jinja2** — Template engine for code generation
- **Standard Library** — Python standard library utilities

**Dependency Direction**: TorchGen is build-time only. ATen and torch/ depend on generated code; TorchGen depends on declarations only.

---

## Trade-Offs and Design Decisions

### Decision 1: Code Generation vs Manual Implementation

**What**: Operation dispatch and bindings are generated from declarations rather than hand-written for each operation.

**Why**:
- Consistency: All operations use same dispatch mechanism
- Maintainability: Update one template, all operations updated
- Scalability: Adding 100 new operations requires only adding 100 lines to YAML, not 1000s lines of C++
- Correctness: Reduces copy-paste errors across 1000+ operations

**Alternatives**:
- Manual Implementation — write dispatch and bindings for each operation (doesn't scale)
- Runtime Generation — generate code at runtime (would be slow)

**Trade-offs**:
- **Pro**: Scalable, maintainable, consistent
- **Con**: Generated code harder to debug; errors in template affect all operations

---

### Decision 2: YAML Declarations as Specification

**What**: Operator behavior specified in human-readable YAML; C++ code generated from YAML.

**Why**:
- Single Source of Truth — YAML is declaration of intent; C++ is implementation detail
- Platform Independence — YAML declarations work on any platform
- Tool Generation — Multiple tools can read YAML and generate different outputs

**Alternatives**:
- C++ Annotations — Embed declarations in C++ headers (less portable; harder to parse)
- Python Definitions — Define operations in Python (requires Python at build time)

**Trade-offs**:
- **Pro**: Portable, maintainable, tool-friendly
- **Con**: Requires YAML parsing; disconnected from C++ implementation

---

### Decision 3: Jinja2 Templates for Code Generation

**What**: Generated C++ code produced by Jinja2 templates applied to YAML declarations.

**Why**:
- Readability: Templates show structure clearly
- Flexibility: Easy to modify without recompiling code generator
- Debugging: Generated code is readable and traceable

**Alternatives**:
- String Concatenation — build code as strings in Python (harder to read)
- Direct AST Manipulation — use C++ AST library (adds complexity)

**Trade-offs**:
- **Pro**: Readable templates, easy to modify
- **Con**: Template syntax can be confusing; Jinja2 dependency

---

## Extension Boundaries

### Extending TorchGen

**Supported Extensions:**
1. **New Operation** — Add declaration to YAML; TorchGen generates code
2. **New Template** — Add new Jinja2 template for different code output
3. **New Backend** — Add backend name to YAML; TorchGen generates backend dispatch

**NOT Supported:**
- Changing how existing declarations are interpreted (breaks backward compatibility)
- Adding non-declarative logic to declarations

### Integration Points

- **YAML Declaration Files** — Native operator definitions
- **Jinja2 Templates** — Code generation templates
- **Python Build System** — Invoked as part of PyTorch build
- **ATen/torch/ Source** — Generated code included in build

---

## Runtime Implications

### Build Time Execution

**Build Sequence:**
1. CMake invokes `python gen.py` with YAML files
2. TorchGen reads YAML declarations
3. Templates applied to generate C++ code
4. Generated code written to source files or headers
5. C++ compiler includes generated code
6. Linking includes generated dispatch logic

**At Runtime:**
- Generated code executes as normal C++ code
- No runtime interpretation of YAML
- Dispatch performs as if hand-written

### Build Artifacts

- **Generated Headers** — `Declarations.h`, `Functions.h`, etc.
- **Generated Bindings** — Python extension code
- **Generated Dispatch** — Operation registration code

### Consistency Considerations

- Generated code must be rebuilt if YAML declarations change
- Build cache invalidation on declaration changes
- Version tracking for generated code

---

## Performance Implications

### Build Time

- **Parsing Time** — O(N) where N = number of operations (~1000)
- **Template Application** — O(N) per template
- **Total Time** — Typically 1-5 seconds for full generation

### Runtime

- **No Impact** — Generated code identical in performance to hand-written equivalent
- **Dispatch Performance** — Generated dispatch as fast as manual dispatch (same logic)

---

## Ownership Boundaries

### What TorchGen Owns

- Code generation logic
- Template design and implementation
- YAML parsing and interpretation
- Consistency checking

### What TorchGen Does NOT Own

- YAML declarations (owned by operation developers)
- Implementation of generated code (implementation is runtime responsibility)
- Backend-specific optimizations

---

## Testing and Validation

### Critical Tests

- **Declaration Parsing** — Verify YAML parsing works correctly
- **Code Generation** — Verify generated code is syntactically valid C++
- **Consistency** — Verify all operations have required backends
- **Binding Generation** — Verify Python bindings compile and work

### Known Gaps

- Limited testing of edge cases (very long operation names, special characters)
- No explicit performance benchmarking of code generation

---

## Related Systems

- **ATen** — Uses generated dispatch code
- **torch/** — Uses generated Python bindings
- **Build System** — Invoked as part of CMake build

---

## References

- `torchgen/gen.py` — Main code generation entry point
- `torchgen/model.py` — YAML parsing and data model
- Native operator declarations (native_functions.yaml, etc.)
