# `aten/src/ATen/native/cpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native/cpu` contains CPU backend kernels for ATen native operators. It provides scalar and SIMD elementwise loops, reductions, pooling, indexing, activation, linear algebra, optimizer, and copy kernels that are registered through native dispatch stubs. Book Chapter 04 calls out this directory as the vectorized CPU kernel layer below TensorIterator and above CPU instruction-set implementations such as Vec256-style `Vectorized<T>` code.

## Key Files

| File | Purpose |
|---|---|
| `Loops.h` | `cpu_kernel`, `cpu_kernel_vec`, contiguous/scalar checks, and explicit vectorized loop templates |
| `BinaryOpsKernel.cpp` | CPU add, multiply, divide, comparison, bitwise, and special-function inner loops registered to stubs |
| `Reduce.h` | Reduction helper templates, vectorized inner and outer reductions, and parallel reduction structure |
| `Activation.cpp` | CPU activation kernels such as threshold, sigmoid, tanh, GELU, and related dispatch registrations |
| `CopyKernel.cpp` | CPU copy and dtype-conversion kernels |
| `README.md` | CPU-kernel authoring guidance for files compiled with CPU feature flags |
| `Intrinsics.h` | CPU intrinsic and vectorization support included by CPU kernels |

## Public Interface

CPU native kernels are not called as user-facing APIs; they register C++ function pointers behind `DispatchStub` names declared in shared native headers and source files. Files in this directory expose functions in `at::native` and `at::native::CPU_CAPABILITY`, and macros such as `REGISTER_DISPATCH` bind a stub like `mul_stub` to a CPU implementation. Kernel authors use `cpu_kernel`, `cpu_kernel_vec`, `binary_kernel_reduce`, `AT_DISPATCH_*` dtype macros, `TensorIteratorBase`, and `Vectorized<T>` from `ATen/cpu/vec/vec.h`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Dispatch stubs, TensorIterator setup, native operator declarations, and shared validation |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Tensor handle, generated op headers, scalar types, and dispatch registration infrastructure |
| [c10/core](c10/core/ADR.md) | depends-on | `ScalarType`, `DeviceType`, scalar wrappers, and tensor metadata |
| [c10/util](c10/util/ADR.md) | depends-on | `irange`, checked loads, type-safe sign math, generic math, and exceptions |
| oneDNN, BLAS, FBGEMM, and platform intrinsics | depends-on | Specialized CPU math libraries and instruction-set code used by selected kernels |

## Runtime Behaviour

Shared native code creates a `TensorIterator` or category-specific arguments, then calls a dispatch stub with `kCPU`; `DispatchStub` selects the CPU function pointer registered by files in this directory. `BinaryOpsKernel.cpp` dispatches on dtype with `AT_DISPATCH_*` macros and executes scalar or vector lambdas through `cpu_kernel` and `cpu_kernel_vec`; for example, multiply uses boolean `&&` for bool tensors, opmath types for reduced floating point, and vectorized multiplication for supported numeric types. `Loops.h` dereferences tensor data through pointer-plus-stride arrays, handles scalar operands with stride zero, runs two vector chunks per loop iteration when possible, and falls back to scalar loops for tails. `Reduce.h` runs reductions through accumulator, combine, and project functions and splits single-output reductions across pieces before combining results.

## Performance Profile

The CPU backend separates operator dispatch from instruction-set dispatch: the dispatcher reaches a CPU stub, then `DispatchStub` chooses default, AVX2, AVX512, SVE, VSX, ZVECTOR, or other compiled variants. `cpu_kernel_vec` uses `Vectorized<T>` loads and stores for contiguous same-type operands and preserves scalar fallback code for non-contiguous strides and remainder elements. `Reduce.h` has specialized inner and outer vectorized reductions, unrolls four vector accumulators, and uses `parallel_for` through TensorIterator for larger workloads. Some kernels deliberately avoid vectorization when the source shows no SIMD benefit, such as integer truncating division in `BinaryOpsKernel.cpp`, which comments that there is no SIMD integer division.

## Design Rationale

Chapter 04 describes CPU kernels as TensorIterator plus vectorized inner loops, and this directory implements that layering. The architecture keeps shape, broadcasting, and dtype rules in shared native code while compiling CPU inner loops with architecture-specific flags that other directories cannot use. `DispatchStub` exists because distribution binaries must choose between several CPU capabilities at runtime. The loop helpers centralize striding, scalar operands, tails, and vector widths so each operator file expresses math rather than reimplementing iteration mechanics.

