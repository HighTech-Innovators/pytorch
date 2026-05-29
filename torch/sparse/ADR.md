# Architecture Decision Record: torch/sparse

## Architectural Role

`torch/sparse` provides sparse tensor support, enabling efficient representation and computation for tensors with many zero elements. Sparse tensors use less memory and can be faster for sparsity-heavy operations, making this module important for certain applications (graph neural networks, recommendation systems).

## Responsibilities

- Implementing sparse tensor formats (COO, CSR, CSC)
- Providing sparse tensor creation and conversion
- Implementing sparse operations (sparse matrix multiplication, element-wise operations)
- Integrating sparse tensors with autograd

## Dependencies

**Inbound**: Research code, GNNs, recommendation systems
**Outbound**: `aten/src/ATen/native` for kernel implementations

## Trade-offs

**Format flexibility vs. operation speed**: Different sparse formats (COO, CSR) are optimal for different operations. Users must select the right format for their workload.

## Runtime Implications

**Format selection**: Users select sparse format based on operations (CSR for SpMV, COO for element-wise).

**Autograd support**: Sparse tensors participate in autograd; gradients of sparse operations are also sparse.

## Performance Implications

**Memory efficiency**: 10-100x memory reduction for high-sparsity tensors.

**Operation speed**: Sparse operations can be 10-100x faster for high-sparsity matrices compared to dense.

## Ownership Boundaries

- **Sparse tensor owns**: compressed data representation
- **Sparse kernels own**: computation

## Verification Points

- `torch/sparse/__init__.py` — Public interface
- `torch/sparse/_compressed.py` — Sparse format implementations
