# `aten/src/ATen/native/cuda`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native/cuda` contains CUDA backend kernels and CUDA library integrations for ATen native operators. A CPU-only build does not compile or execute these kernels, but the source defines the CUDA architecture for elementwise kernels, reductions, BLAS, FFT, pooling, convolution, indexing, and sparse-adjacent GPU operations. Book Chapter 04 places this directory beside the CPU backend as the CUDA implementation layer behind `native_functions.yaml` dispatch entries and shared TensorIterator setup.

## Key Files

| File | Purpose |
|---|---|
| `CUDALoops.cuh` | GPU elementwise loop helpers, scalar CPU operand handling, vectorized policies, and launch configuration templates |
| `AbsKernel.cu` | CUDA absolute-value kernel using `gpu_kernel`, Jiterator for complex types, dtype dispatch, and `abs_stub` registration |
| `BinaryMulKernel.cu` | CUDA multiplication kernel using TensorIterator, opmath functors, scalar handling, and `mul_stub` registration |
| `Blas.cpp` | CUDA BLAS-backed matrix and vector operation helpers, stride preparation, activation epilogues, and tunable GEMM integration |
| `CuFFTPlanCache.h` | cuFFT parameter hashing, layout embedding, handle lifetime, and plan-cache data structures |
| `Copy.cu` | CUDA copy and conversion kernels |
| `CUDAJitLoops.cuh` | Jiterator-backed CUDA loop support for generated device code |

## Public Interface

CUDA native kernels register function pointers behind shared native stubs and generated dispatcher entries; user code reaches them through ordinary ATen and Python operators. Kernel files expose `at::native` functions such as `abs_kernel_cuda` and `mul_kernel_cuda`, then bind them with `REGISTER_DISPATCH`. CUDA loop authors use `gpu_kernel`, `gpu_kernel_with_scalars`, `opmath_symmetric_gpu_kernel_with_scalars`, Jiterator helpers, `TensorIteratorBase`, CUDA dispatch macros, and `c10::cuda` guard and stream utilities.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Native operator declarations, TensorIterator setup, dispatch stubs, and shared validation |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Tensor handle, generated op headers, scalar metadata, and dispatcher registration infrastructure |
| [c10/cuda](c10/cuda/ADR.md) | depends-on | CUDA guards, streams, math compatibility, launch checks, and allocator/runtime state |
| [c10/core](c10/core/ADR.md) | depends-on | `ScalarType`, `Device`, `DispatchKey`, and tensor metadata |
| CUDA, cuBLAS, cuFFT, and ROCm/HIP equivalents | depends-on | GPU runtime, BLAS, FFT, and architecture-specific compilation targets |

## Runtime Behaviour

Shared native code reaches this directory through dispatch stubs or generated CUDA dispatch entries. `AbsKernel.cu` reads `iter.dtype()`, chooses complex Jiterator or direct `gpu_kernel` paths, dispatches across half, bfloat16, bool, and other scalar types, and registers `abs_kernel_cuda` to `abs_stub`. `BinaryMulKernel.cu` reads `iter.common_dtype()`, uses a complex-half Jiterator path when enabled, otherwise launches an opmath GPU functor with scalar support and registers `mul_kernel_cuda` to `mul_stub`. `CUDALoops.cuh` builds policies for GPU elementwise kernels, lifts one CPU scalar into a kernel parameter for expressions such as `cuda_tensor + 5`, and requires all remaining operands and outputs to live on the GPU.

## Performance Profile

`CUDALoops.cuh` computes element counts per thread from input/output type sizes, uses `elementwise_thread_work_size`, and selects vectorized memory policies for full blocks with scalar-unrolled fallback for remainders. On CUDA, vector size 8 is compiled for SM90 and SM10x paths to limit binary size while preserving wide vectorized loads on architectures that use them. `Blas.cpp` prepares matrices for cuBLAS by inspecting strides, preserving usable leading dimensions, resolving conjugation, and cloning to contiguous storage only when layout fails BLAS requirements. `CuFFTPlanCache.h` hashes signal sizes, input/output strides, transform type, and value type into `CuFFTParams`, embeds strides when possible, and marks layouts that require cloning before execution.

## Design Rationale

CUDA kernels share ATen schemas and TensorIterator semantics with CPU kernels, but they need GPU launch policies, stream state, scalar lifting, and vendor libraries that belong in a separate backend directory. Chapter 04 emphasizes backend-independent declarations in `native_functions.yaml`; this directory fulfills the CUDA side of those declarations without changing operator schemas. Elementwise helpers centralize block sizing, vectorization, scalar operands, and Jiterator paths so individual kernels only provide math functors. Library integrations such as cuBLAS and cuFFT live here because their performance depends on CUDA-specific layout, stream, and workspace constraints.

