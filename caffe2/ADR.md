# ADR: caffe2 — Legacy Deep Learning Framework

**Status**: MAINTENANCE  
**Last Updated**: 2026-05-28

## Architectural Role

Caffe2 is a legacy deep learning framework now integrated into PyTorch. It provides:

- **Operator definitions**: Legacy operator specifications predating PyTorch API
- **Protobuf serialization**: Binary format for model persistence
- **Legacy training loops**: Pre-PyTorch training code (mostly deprecated)
- **Backwards compatibility**: Support for models trained with original Caffe2

caffe2 is classified as **Legacy** (maintenance mode) because most functionality is superseded by torch/. It is maintained for backwards compatibility only.

## Responsibilities

### What caffe2 Owns
- **Legacy operators**: Operators not yet migrated to torch API
- **Protobuf model format**: Serialization and deserialization
- **Legacy training code**: Training routines predating modern PyTorch
- **Backwards compatibility**: Support for old Caffe2 models

### What caffe2 Does Not Own (delegated to torch/)
- **Modern deep learning API**: Implemented by torch.nn
- **Modern training loops**: Implemented by torch.optim
- **Automatic differentiation**: Implemented by torch.autograd

## Dependencies

### Internal Dependencies (caffe2 → other modules)
- **ATen**: Uses tensor operations
- **c10**: Uses core abstractions

### External Dependencies (other modules → caffe2)
- **Minimal**: torch/ provides wrapper functions for compatibility
- **Legacy code only**: New code should not depend on caffe2

## Trade-offs and Design Decisions

### 1. Backwards Compatibility vs Code Cleanup
**Decision**: Maintain Caffe2 codebase for backwards compatibility; recommend migration to torch/  
**Rationale**:
- **User migration path**: Existing Caffe2 users can port to torch incrementally
- **Minimal breakage**: No forced migrations of working code

**Trade-off**: Legacy code clutters codebase; maintenance burden

## Notes and Caveats

1. **Caffe2 is in maintenance mode**: New features not added; existing code maintained for compatibility
2. **No performance optimizations**: Caffe2 operators use generic implementations; torch operators are optimized
3. **Protobuf format is immutable**: Model format cannot change; breaking compatibility is not acceptable
4. **Migration to torch is recommended**: New projects should use torch.nn, not caffe2
