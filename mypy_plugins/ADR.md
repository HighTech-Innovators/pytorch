# ADR: mypy_plugins — Type Checking Extensions

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The mypy_plugins directory contains Python-based extensions for the mypy static type checker that enhance type inference capabilities for PyTorch code. The directory provides custom type plugins that handle PyTorch's dynamic type system by extending mypy's core type inference logic to understand tensor operations, device placement, and datatype conversions that would otherwise be opaque to a static analyzer. mypy_plugins bridges the semantic gap between PyTorch's runtime type flexibility and the strict type constraints of Python's static type system, enabling developers to write type-safe PyTorch code while preserving the framework's dynamic strengths. mypy_plugins is classified as **Development Support** (non-runtime) because it provides tooling for developers, not core functionality.

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
The mypy_plugins module requires **mypy** (the static type checker framework) and **Python 3.6+** to develop and test custom type plugins. The plugin system is implemented in Python and loaded by mypy at analysis time via the plugin discovery mechanism in mypy's configuration system (typically `mypy.ini` or `pyproject.toml`).

### Runtime Dependencies
No runtime dependencies exist; mypy_plugins operates exclusively during development at static analysis time. The plugins are only invoked when developers run mypy type checking on their codebase; they do not affect runtime behavior or application deployment.

## Trade-offs and Design Decisions

### 1. Static Types for Dynamic Language
**Decision**: Provide mypy plugins to enable static type checking for PyTorch code  
**Rationale**: 
PyTorch's dynamic tensor operations and method overloading create challenges for static type checkers like mypy, which are designed for languages with fixed type signatures. By implementing custom type plugins, PyTorch enables developers to catch type errors before runtime through IDE integration and CI/CD type checking, improving code reliability without sacrificing PyTorch's runtime flexibility. The plugin approach delegates type inference extensions to mypy's own plugin architecture rather than forking or patching mypy itself.

**Trade-off**: Type hints are optional in PyTorch code and not enforced at runtime; incorrect type hints do not cause runtime failures, only static analysis failures. This means type safety is aspirational rather than guaranteed. Additionally, maintaining plugin compatibility with each new mypy release imposes ongoing maintenance burden (each mypy release may change plugin APIs). The benefit—early error detection for type-aware developers—justifies this burden, but it means projects not using mypy type checking gain no benefit from mypy_plugins.

## Extension Boundaries

### Public Extension Points
1. **Custom type plugins**: Extend mypy for domain-specific type inference

## Notes and Caveats

1. **Type hints are optional**: Not all PyTorch code is type-checked
2. **Dynamic typing limitations**: PyTorch's dynamic nature makes static typing challenging
3. **Type stub maintenance**: .pyi files must be kept in sync with runtime implementations
