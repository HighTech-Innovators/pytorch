# Architecture Decision Record: torchgen

## Architectural Role

`torchgen` is the code generation infrastructure for PyTorch operators. It takes operator definitions (in ATen native_functions.yaml) and generates C++ operator registrations, Python bindings, and type signatures. This reduces manual coding and ensures consistency across operator implementations.

## Responsibilities

- Parsing operator definitions (YAML format)
- Generating C++ dispatcher registrations
- Generating Python bindings
- Generating type signatures and documentation

## Dependencies

**Inbound**: ATen operator definition files
**Outbound**: Generated C++ and Python files

## Trade-offs

**Code generation vs. manual coding**: Generated code is verbose but consistent and automatically updated when operator definitions change.

## Runtime Implications

**Build-time code generation**: Torchgen is run during build, not runtime.

## Ownership Boundaries

- **Torchgen owns**: code generation logic
- **Generated files own**: static artifacts (produced files)

## Verification Points

- `torchgen/` — Implementation directory
- `aten/src/ATen/native_functions.yaml` — Operator definitions
