# `aten/src/ATen/native/cpu`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native/cpu` holds the vectorized CPU kernel bodies for element-wise and reduction operators. These files are compiled multiple times — once per CPU instruction set (default/AVX2/AVX-512) — and register their variants into the dispatch stubs declared in `aten/src/ATen/native`. This is the directory that turns a `TensorIterator` range into actual SIMD arithmetic on CPU.

## Key Files

| File | Purpose |
|---|---|
| `aten/src/ATen/native/cpu/Loops.h` | `cpu_kernel`, `cpu_kernel_vec`, `basic_loop`, `vectorized_loop` — the CPU loop drivers |
| `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` | Vectorized `add`/`mul`/`sub`/… kernels registered to binary stubs |
| `aten/src/ATen/native/cpu/Activation.cpp` | Vectorized ReLU, GELU, sigmoid, tanh kernels |
| `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp` | Vectorized reduction kernels (`sum`, `mean`, …) |
| `aten/src/ATen/native/cpu/Reduce.h` | Shared reduction loop templates |
| `aten/src/ATen/native/cpu/ReduceUtils.h` | Helpers for reduction accumulation and vectorization |

## Public Interface

These are internal kernel registrations, not a public API. Entry points: `cpu_kernel(iter, scalar_lambda)`, `cpu_kernel_vec(iter, scalar_lambda, vec_lambda)`, `basic_loop(...)`, `vectorized_loop(...)`, and the `REGISTER_DISPATCH` / `REGISTER_AVX2_DISPATCH` / `REGISTER_AVX512_DISPATCH` macros that bind these kernels to stubs such as `add_stub`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | `TensorIteratorBase` ranges, `AT_DISPATCH_*`, dispatch-stub macros |
| `aten/src/ATen/cpu/vec/` | depends-on | `at::vec::Vectorized<T>` SIMD primitives (AVX2/AVX-512/NEON) |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depended-on-by | Declares the stubs these kernels register into |

## Runtime Behaviour

A binary op kernel calls `cpu_kernel_vec(iter, scalar_fn, vec_fn)`; `Loops.h` iterates the `TensorIterator`'s prepared ranges, using `basic_loop` for the scalar tail and `vectorized_loop` (over `at::vec::Vectorized<T>`) for the SIMD-eligible contiguous inner dimension. Each source file is compiled several times under different `-mavx2` / `-mavx512` flags, producing separate function symbols; at process startup the runtime CPU-feature detector selects the fastest registered variant for each stub. Whether the vectorized path is taken depends on the iterator's contiguity and dtype decisions made upstream in `TensorIterator.cpp`.

## Performance Profile

This directory is where CPU compute-bound element-wise performance is won or lost. When a tensor is contiguous and large, `vectorized_loop` processes multiple elements per instruction via `Vectorized<T>`, and the AVX-512 variant delivers the widest lanes; badly-strided or tiny tensors fall back to `basic_loop` scalar iteration with much lower throughput. Reductions in `ReduceOpsKernel.cpp` are memory-bandwidth bound and use vectorized accumulators to keep the ALU fed. There is no allocation in these loops themselves — outputs are pre-allocated by the iterator — so the cost is purely arithmetic and memory traffic, making memory layout the dominant tunable.

## Design Rationale

Multi-versioned compilation (one kernel source, N instruction-set builds) lets a single binary run optimally on old and new CPUs without source duplication, selected cheaply at startup rather than per call. Keeping the scalar and vector lambdas together in `cpu_kernel_vec` lets one kernel express both the SIMD fast path and the correct scalar tail, and lets the compiler auto-vectorize the scalar path too. Building on the shared `at::vec::Vectorized<T>` abstraction keeps kernels ISA-agnostic in source while still emitting architecture-specific SIMD. This directory is the primary optimization surface for the CPU-only workloads targeted by this project.
