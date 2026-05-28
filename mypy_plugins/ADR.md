# ADR: mypy_plugins — Type Checking Extensions

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

mypy_plugins provides type checking extensions for PyTorch code:

- **Custom type plugins**: mypy plugins for type inference
- **Tensor type hints**: Type information for dynamic Tensor operations
- **Device type inference**: Static detection of device placement

mypy_plugins is classified as **Development Support** (non-runtime) because it provides tooling for developers, not core functionality.

## Responsibilities

### What mypy_plugins Owns
- **Custom mypy plugins**: Extend mypy type checker for PyTorch
- **Type inference rules**: Handle dynamic typing in tensor operations
- **Device inference**: Infer tensor device from context

### What mypy_plugins Does Not Own
- **Type definitions**: torch.pyi files define type stubs
- **Runtime type checking**: torch.jit handles runtime types

## Dependencies

### Build-Time Dependencies
- **mypy**: Type checker framework
- **Python**: Plugin is Python-based

### Runtime Dependencies
- None; plugins are used at development time (static analysis)

## Trade-offs and Design Decisions

### 1. Static Types for Dynamic Language
**Decision**: Provide mypy plugins to enable static type checking for PyTorch code  
**Rationale**:
- **Type safety**: Catch type errors before runtime
- **IDE support**: Enable autocomplete and type hints in IDEs

**Trade-off**: Type hints are optional; not enforced at runtime; optional for users

## Extension Boundaries

### Public Extension Points
1. **Custom type plugins**: Extend mypy for domain-specific type inference

## Notes and Caveats

1. **Type hints are optional**: Not all PyTorch code is type-checked
2. **Dynamic typing limitations**: PyTorch's dynamic nature makes static typing challenging
3. **Type stub maintenance**: .pyi files must be kept in sync with runtime implementations
