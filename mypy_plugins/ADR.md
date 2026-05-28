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
1. **Custom type plugins**: Add new mypy plugin in `mypy_plugins/` directory implementing `mypy.plugin.Plugin` interface
2. **Type inference rules**: Extend `mypy_plugins/` handlers to recognize new tensor method patterns or operator overloads
3. **Device type inference**: Add inference rules in appropriate plugin module for device-specific type narrowing

### Plugin Development Pattern
To add a new type plugin for PyTorch constructs:
1. Create plugin class inheriting from `mypy.plugin.Plugin` in `mypy_plugins/<feature>_plugin.py`
2. Implement plugin hook methods (e.g., `get_class_hook()`, `get_decorator_hook()`)
3. Register plugin in `mypy.ini` or `pyproject.toml` under `[mypy]` section with `plugins = mypy_plugins.<feature>_plugin`
4. Add type stubs in `torch/*.pyi` files for runtime method signatures
5. Test with `mypy --config-file=mypy.ini tests/` to verify type inference works

## Performance Implications

### Static Analysis Overhead
- **Type checking cost**: Mypy plugins add overhead to static analysis (~5-10% per plugin), not runtime
- **Iteration cost**: Developers using mypy see slower IDE type checking (~1-2 seconds per save)
- **CI/CD impact**: Type checking adds ~30-60 seconds to CI pipeline per Python codebase scan
- **Caching**: Mypy caches analysis results in `.mypy_cache/` to avoid redundant checks

### No Runtime Impact
- Mypy plugins are loaded only during static analysis; compiled Python code contains no type information
- Runtime behavior is unchanged; type hints are comments only (in Python 3.5+, type hints are metadata, not executable)

## Ownership Boundaries

### mypy_plugins Owns
- **Plugin implementation**: Custom type inference rules in plugin modules
- **Type inference extensions**: Logic to understand PyTorch-specific type patterns (device inference, dtype narrowing)
- **Plugin registration**: Configuration to load plugins at mypy startup

### mypy_plugins Borrows
- **Type stubs (.pyi files)**: Defined separately in `torch/*.pyi` (not in mypy_plugins/)
- **mypy framework**: Uses mypy.plugin.Plugin API (maintained by mypy project, not PyTorch)
- **Type definitions**: Inherits typing from Python's typing module and torch type stubs

## Key Files and References

| File | Purpose |
|---|---|
| `mypy_plugins/` | Root plugin directory |
| `mypy_plugins/__init__.py` | Plugin discovery entry point |
| `mypy_plugins/tensor_plugin.py` (example) | Custom type inference for Tensor.to(), .cpu(), .cuda() methods |
| `torch/*.pyi` | Type stub files defining method signatures |
| `mypy.ini` | Configuration file registering mypy plugins |
| `pyproject.toml` | Alternative configuration format for mypy plugins section |
| `tests/` | Type checking tests (pytest with `--mypy` flag) |

## Runtime Constraints

### Type Information Availability
- Type hints are present at runtime as `__annotations__` on functions/classes, but mypy plugins cannot access this (mypy is a static tool)
- Plugin decisions cannot be informed by runtime values (e.g., cannot inspect actual tensor device at type-check time)
- Inference must be conservative: when type is ambiguous, assume most general type (e.g., `Union[Tensor]` if device unknown)

### Compatibility with Dynamic Code
- PyTorch's dynamic nature (runtime method creation, monkey-patching) is invisible to mypy
- Type plugins cannot track dynamic attributes; they work only with statically analyzable code
- This limitation is acceptable because static type checking is optional; dynamic code works at runtime

## Notes and Caveats

1. **Type hints are optional**: Not all PyTorch code requires type hints; untyped code bypasses mypy_plugins entirely
2. **Dynamic typing limitations**: PyTorch's dynamic tensor operations (e.g., `tensor.view(-1, 10)` shape inference) cannot be statically typed; plugins use `Any` for unknowable types
3. **Type stub maintenance burden**: `.pyi` files must be manually maintained and kept in sync with runtime implementations as new methods are added
4. **Plugin API churn**: MyPy's plugin API evolves; plugins may break on major mypy version upgrades (requires maintenance)
5. **Inheritance complexity**: PyTorch method resolution order (MRO) with operator overloading creates complex type inference scenarios; plugins must handle these carefully
