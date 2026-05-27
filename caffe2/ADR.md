# Architecture Decision Record: caffe2 (Legacy Framework Support)

## Architectural Role

`caffe2/` is **PyTorch's legacy framework compatibility layer**, preserving Caffe2 support for backward compatibility. It provides:

1. **Caffe2 operators**: Port of Caffe2-specific operations
2. **Model conversion**: Tools for migrating Caffe2 models to PyTorch
3. **API compatibility**: Maintaining Caffe2-like interfaces
4. **Serialization**: Loading/saving Caffe2 models

Key insight: `caffe2/` is **backward compatibility layer** from PyTorch's merger with Caffe2. Most new development uses torch instead.

## Responsibilities

### What This Subsystem Owns

1. **Caffe2 Operators** (`caffe2/core/`)
   - Caffe2-specific operation implementations
   - Operator registry for Caffe2 API

2. **Core Infrastructure** (`caffe2/core/`)
   - Workspace concept (Caffe2's execution model)
   - Operator scheduling

3. **Utilities** (`caffe2/utils/`)
   - Helper functions for Caffe2 models
   - Performance utilities

### What This Subsystem Does NOT Own

- New operation development (use torch)
- Automatic differentiation (use torch.autograd)
- Modern model development (use torch.nn)

## Dependencies

### Upstream Dependencies

- Legacy Caffe2 model users
- Migration scenarios

### Downstream Dependencies

- PyTorch core (torch, aten)
- ONNX export (for Caffe2 model export)

## Trade-offs and Design Decisions

### Legacy Support vs. Deprecation

**Decision**: Maintain Caffe2 API for backward compatibility, but discourage new use.

**Trade-off**:
- ✅ **Advantage**: Existing models still work
- ✅ **Advantage**: Gradual migration path
- ❌ **Disadvantage**: Maintenance burden
- ❌ **Disadvantage**: Code complexity

**Evidence**: Caffe2 operators coexist with torch operators.

## Key Implementation Files

| File | Purpose |
|---|---|
| `caffe2/core/` | Core Caffe2 infrastructure |
| `caffe2/utils/` | Utilities |

Last Verified: 2026-05-27
