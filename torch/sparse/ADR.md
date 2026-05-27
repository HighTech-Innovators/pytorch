# Architecture Decision Record: torch.sparse (Sparse Tensor Support)

## Architectural Role

`torch.sparse` is **PyTorch's sparse tensor infrastructure**, providing efficient storage and computation for tensors with few non-zero elements. It enables:

1. **Sparse storage formats**: COO, CSR, CSC, BSR, BSC formats
2. **Sparse operations**: Operations optimized for sparse layout
3. **Memory efficiency**: Storing only non-zero values
4. **Specialized kernels**: Device-specific sparse implementations
5. **Interoperability**: Converting between dense and sparse

Key insight: `torch.sparse` is **alternative tensor layout** with specialized operations. Sparse tensors use different storage (indices + values) enabling memory-efficient representation and computation when sparsity is high.

## Responsibilities

### What This Subsystem Owns

1. **Sparse Tensor Representations**
   - COO (Coordinate) format
   - CSR (Compressed Sparse Row) format
   - Other formats (CSC, BSR, etc.)

2. **Sparse Operations**
   - sparse-dense matrix multiply
   - sparse-sparse operations
   - Reductions and aggregations

3. **Layout Conversion**
   - Dense to sparse conversion
   - Sparse to dense conversion
   - Format conversion (COO to CSR, etc.)

4. **Sparse-Specific Kernels**
   - CPU sparse operations
   - CUDA sparse operations (via cuSPARSE)

### What This Subsystem Does NOT Own

- **Dense tensor operations**: torch and torch.nn own this
- **Autograd integration**: torch.autograd handles gradients
- **Device management**: torch.device owns this

## Dependencies

- **Upstream**: Users needing sparse computation
- **Downstream**: ATen kernels, cuSPARSE library

## Trade-offs and Design Decisions

### Multiple Formats vs. Single Format

**Decision**: Support multiple sparse formats.

**Trade-off**:
- ✅ **Advantage**: Each format optimal for different access patterns
- ✅ **Advantage**: Users can choose format
- ❌ **Disadvantage**: Implementation complexity
- ❌ **Disadvantage**: Multiple APIs to learn

**Evidence**: COO, CSR, CSC formats supported.

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/sparse/` | Sparse tensor implementation |

Last Verified: 2026-05-27
