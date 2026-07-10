# `aten/src/ATen/native`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native` holds the concrete implementations of PyTorch's ~2000 operators — the actual kernel bodies for arithmetic, reductions, matrix ops, convolutions, normalization, activations, indexing, and shape operations. It is where the mathematical work of the forward pass is defined and registered against the dispatcher.

## Key Files

| File | Purpose |
|---|---|
| `aten/src/ATen/native/native_functions.yaml` | Declarative registry of operator signatures and per-key dispatch intent |
| `aten/src/ATen/native/BinaryOps.cpp` | Element-wise binary ops (`add`, `mul`, …) built on `TensorIterator` + stubs |
| `aten/src/ATen/native/LinearAlgebra.cpp` | `mm`, `addmm`, `matrix_power` — BLAS delegation and composites |
| `aten/src/ATen/native/Convolution.cpp` | Convolution front-end; dispatches to MKLDNN/native backends |
| `aten/src/ATen/native/SoftMax.cpp` | Softmax/log-softmax reductions |
| `aten/src/ATen/native/layer_norm.cpp` | Layer normalization (reduction + broadcast) |
| `aten/src/ATen/native/Activation.cpp` | ReLU, GELU, sigmoid, tanh, … |
| `aten/src/ATen/native/ReduceOps.cpp` | `sum`, `mean`, `max`, `norm` reductions |
| `aten/src/ATen/native/cpu/` | Vectorized CPU kernel bodies (see child ADR) |

## Public Interface

Operators are exposed through generated `at::` functions rather than direct calls into this directory. Internal registration entry points: `TORCH_META_FUNC(op)`, `TORCH_IMPL_FUNC(op_out_cpu)`, direct kernels like `add_cpu()`, dispatch stubs (`DECLARE_DISPATCH`/`REGISTER_DISPATCH`, e.g. `add_stub`), and composite implementations that call other `at::` ops.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | `TensorIterator`, `Dispatch.h`, `Parallel.h`, dispatcher registration |
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl`, `ScalarType`, `MemoryFormat` |
| [aten/src/ATen/native/cpu](aten/src/ATen/native/cpu/ADR.md) | depends-on | Vectorized CPU stub implementations |
| MKL / OpenBLAS / MKLDNN | depends-on | Vendor libraries for matmul, convolution, FFT |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depended-on-by | Generated backward nodes call these forward kernels' derivatives |

## Runtime Behaviour

Kernels follow three patterns: direct kernels (e.g. `add_cpu` builds a `TensorIterator` via `borrowing_binary_op` then calls `add_stub(kCPU, iter, alpha)`), structured kernels (a `TORCH_META_FUNC` runs device-agnostic shape inference once, then a per-backend `TORCH_IMPL_FUNC` computes), and `CompositeExplicitAutograd` kernels (e.g. `matrix_power`) implemented purely by calling other dispatched `at::` ops so they work on any backend. Dispatch stubs (`add_stub`) add a second, orthogonal level of selection: at runtime the CPU feature detector picks the AVX2/AVX-512/scalar variant registered via `REGISTER_AVX2_DISPATCH` and friends.

## Performance Profile

Element-wise kernels are thin lambdas over `TensorIterator` ranges, so their cost is dominated by iterator setup and memory layout rather than the arithmetic itself; a broadcast over a non-contiguous tensor forces stride computation and reordering before the loop. Reductions (`SoftMax.cpp`, `layer_norm.cpp`, `ReduceOps.cpp`) are memory-bandwidth bound, scanning the reduced dimension. Matrix ops and convolutions delegate to MKL/OpenBLAS/MKLDNN, moving the hot path into compute-bound, SIMD-optimized vendor code. The structured-kernel `meta` function pre-allocates outputs, a per-call allocation site for non-in-place ops.

## Design Rationale

Splitting operator implementations across many focused `.cpp` files (rather than one monolith) lets each operator family evolve independently while sharing `TensorIterator` and dispatch infrastructure. The structured-kernel pattern separates shape logic from computation so shape errors surface before any kernel runs and the `Meta` key can trace shapes without allocating data. `CompositeExplicitAutograd` operators avoid per-backend duplication for ops expressible in terms of primitives. Delegating matmul/conv to vendor BLAS/MKLDNN is the pragmatic choice: those libraries are hand-tuned far beyond what a generic kernel could achieve. CUDA and other backend kernels coexist in sibling directories but are inert in this CPU-only deployment.
