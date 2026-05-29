# Architecture Decision Record: torch/linalg

## Architectural Role

`torch/linalg` provides linear algebra operations (QR, SVD, eigendecomposition, solving linear systems, matrix inversion) using optimized BLAS/LAPACK backends. These are core numerical operations for scientific computing, physics simulations, and specialized machine learning algorithms. The module is important for numerical computing but not Runtime Critical for standard deep learning.

## Responsibilities

- Implementing linear algebra operators (torch.linalg.qr, torch.linalg.svd, torch.linalg.eig, torch.linalg.solve, torch.linalg.inv)
- Integrating with BLAS libraries (cuBLAS for GPU, MKL for CPU)
- Supporting matrix decompositions and factorizations
- Providing dtype flexibility (float32, float64, complex64, complex128)
- Implementing batched operations for efficiency

## Dependencies

**Inbound** (what depends on torch/linalg):
- Scientific computing code
- Optimization algorithms (SVD-based pseudoinverse, QR-based least squares)
- Physics simulations and neural ODEs
- Specialized neural networks (orthogonal networks, etc.)

**Outbound** (what torch/linalg depends on):
- BLAS libraries (cuBLAS via CUDA, MKL via CPU)
- `aten/src/ATen/native` for kernel implementation
- `c10/core` for tensor abstractions

## Trade-offs

**cuBLAS/MKL wrapper vs. custom implementations**: Using vendor libraries trades implementation simplicity for production-grade performance. Custom implementations would be harder to maintain but might offer specialized optimizations.

**Batched operations overhead**: Supporting batched linear algebra adds complexity to dispatcher logic but enables efficient multi-matrix operations without explicit Python loops.

## Extension Boundaries

- **Custom decompositions**: New linear algebra operations can be implemented by composing existing primitives.
- **Custom backends**: Advanced users can register alternative BLAS implementations.

## Runtime Implications

**Backend selection**: CPU operations use MKL (or fallback LAPACK), GPU operations use cuBLAS/cuSOLVER.

**In-place operations**: Some operations support in-place variants (e.g., `torch.linalg.inv_ex`) for memory efficiency.

**Error handling**: Singular matrices and numerical errors raise exceptions or return status codes depending on operation.

## Performance Implications

**BLAS library overhead**: cuBLAS and MKL calls add 10-50 microseconds overhead but provide 10-1000x faster computation compared to naive Python implementations.

**Memory efficiency**: Batched operations process multiple matrices efficiently, avoiding Python loop overhead.

**Numerical stability**: Vendor libraries are highly tuned for numerical stability, reducing risk of degradation due to accumulation errors.

## Ownership Boundaries

- **torch.linalg owns**: operator interface and batching logic
- **BLAS libraries own**: actual linear algebra computation
- **Tensor owns**: matrix data storage

## Verification Points

- `torch/linalg/__init__.py` — Public API interface
- `torch/linalg/` — Implementation directory
- `aten/src/ATen/native/linalg/` — Kernel implementations
