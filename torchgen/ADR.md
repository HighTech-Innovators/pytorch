# Architecture Decision Record: torchgen (Code Generation for Operators)

## Architectural Role

`torchgen` is **PyTorch's operator code generation system**, automatically generating C++ and Python bindings, dispatch code, and documentation from declarative operator schemas. It enables:

1. **Single source of truth**: Operator definitions in YAML drive code generation
2. **Consistency**: Ensuring all variants (function, method, inplace) are consistent
3. **Dispatcher registration**: Automatically registering operations with the dispatcher
4. **Backend abstraction**: Generating dispatch logic for device backends
5. **Python binding generation**: Creating pybind11 bindings from schemas

Key insight: `torchgen` is **schema-driven code generation**. Rather than manually writing dispatcher code, kernel wrappers, and bindings, operators are declared once in YAML, and torchgen generates all the supporting infrastructure. This reduces duplication and ensures consistency.

## Responsibilities

### What This Subsystem Owns

1. **Schema Parsing** (`torchgen/api/*.py`)
   - Reading `native_functions.yaml` definitions
   - Parsing operation signatures and metadata
   - Extracting device dispatch information

2. **Code Generation** (`torchgen/gen.py` and backends)
   - Generating C++ dispatcher code
   - Generating pybind11 binding code
   - Generating wrapper functions
   - Generating tests

3. **Backend Selection** (`torchgen/code_generators/`)
   - Generating dispatch tables
   - Backend-specific kernel lookup code
   - Fallback chain generation

4. **Documentation** (`torchgen/`)
   - Auto-generating operation documentation
   - Generating type hints for Python
   - Creating signatures for IDE autocomplete

5. **Variant Generation**
   - Generating in-place variants (add vs. add_)
   - Generating out-of-place variants (add vs. add.out)
   - Consistent naming conventions

### What This Subsystem Does NOT Own

- **Operation implementations**: Individual backend kernels (CPU, CUDA, etc.)
- **Operator execution**: ATen dispatcher executes operations
- **Python bindings**: pybind11 library handles binding implementation
- **Graph building**: torch.fx and related modules
- **Operation optimization**: Individual kernels are optimized separately

## Dependencies

### Upstream Dependencies (What Uses This)

- **PyTorch build system**: Runs torchgen during compilation
- **Developers adding new operations**: Write YAML schema for new ops
- **Operator framework**: Consumes generated code

### Downstream Dependencies (What This Uses)

- **native_functions.yaml**: Source of operator definitions
- **Backend kernel definitions**: Links generated code to implementations
- **Python pybind11**: Library for Python bindings
- **C++ compiler**: Compiles generated code

### Dependency Direction

```
native_functions.yaml (operator definitions)
    ↓
torchgen (code generation)
    ├─→ Parse schemas
    ├─→ Generate dispatcher code
    ├─→ Generate bindings
    └─→ Generate documentation
        ↓
    Generated C++ code
    Generated Python bindings
    Generated documentation
```

## Trade-offs and Design Decisions

### Schema-Driven vs. Manual Code

**Decision**: Define operators in YAML; generate supporting code.

**Trade-off**:
- ✅ **Advantage**: Single source of truth; less duplication
- ✅ **Advantage**: Consistency across all operation variants
- ✅ **Advantage**: Easy to add new backends or variants
- ❌ **Disadvantage**: Complex generation logic; hard to understand generated code
- ❌ **Disadvantage**: Slow iteration (must regenerate after each schema change)

**Evidence**: `aten/src/ATen/native/native_functions.yaml` is the schema source.

### Structured Delegates for Fallback

**Decision**: Use `structured_delegate` to specify fallback implementations.

**Trade-off**:
- ✅ **Advantage**: Easy to implement fallback without writing dispatcher code
- ✅ **Advantage**: Supports composition (add.out implemented as add + write)
- ❌ **Disadvantage**: Requires understanding delegation concept
- ❌ **Disadvantage**: May hide performance issues

**Evidence**: `add.Tensor` delegates to `add.out` for structured implementation.

### Variants for Language Idioms

**Decision**: Generate different variants for different language idioms (function vs. method, inplace vs. new).

**Trade-off**:
- ✅ **Advantage**: Pythonic API (can write `a.add(b)` or `torch.add(a, b)`)
- ✅ **Advantage**: Efficient computation (in-place operations save allocation)
- ❌ **Disadvantage**: Code generation complexity
- ❌ **Disadvantage**: More variants mean more maintenance burden

**Evidence**: `add`, `add_`, `add.out` generated from single schema.

## Runtime Implications

### Build-Time Impact

1. **Schema parsing**: Reading and validating YAML (fast, seconds)
2. **Code generation**: Generating C++ and Python code (varies, can be minutes)
3. **Compilation**: Compiling generated C++ code (slow, can be hours)
4. **Testing**: Running generated tests

### Generated Code Characteristics

- **Dispatcher code**: Contains kernel lookup tables; O(1) dispatch lookup
- **Binding code**: pybind11 wrappers for Python-C++ boundary
- **Documentation**: Auto-generated from schema metadata

## Performance Implications

### Build-Time vs. Runtime

- **Build-time**: Significant generation and compilation time
- **Runtime**: Generated code has no overhead vs. hand-written (after compilation)
- **Caching**: Generated code cached; no regeneration unless schema changes

## Key Implementation Files

| File | Purpose |
|---|---|
| `torchgen/gen.py` | Main code generation entry point |
| `torchgen/api/` | Schema parsing and AST |
| `torchgen/code_generators/` | Backend-specific code generation |
| `aten/src/ATen/native/native_functions.yaml` | Operator schema definitions |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Examined `torchgen/` structure and code generation pipeline
2. **Book References**: Chapter 4 (Operators) explains schema-based generation
3. **Code Flow**: Understanding how YAML schemas become generated code

Last Verified: 2026-05-27
