# `torch/_inductor/codegen`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_inductor/codegen` emits executable source and wrapper code from scheduled Inductor IR, as mapped in book chapter 08. It contains backend-specific schedulers and kernels for Triton GPU code, C++/OpenMP CPU code, C++ wrappers, Python wrappers, multi-kernel selection, device overrides, and specialized templates. The directory translates loop bodies, symbolic indexing expressions, buffer arguments, workspace requirements, reductions, vectorization decisions, and kernel calls into runnable modules that `codecache.py` compiles and caches.

## Key Files

| File | Purpose |
|---|---|
| `common.py` | Shared codegen abstractions: `Kernel`, `KernelArgs`, CSE, symbolic printers, backend registry, workspace args, and wrapper/backend feature definitions |
| `triton.py` | Triton source generation, Triton scheduling, block/tile selection, reduction helpers, metadata, and async compilation hooks |
| `cpp.py` | C++ kernel generation for CPU loops, OpenMP reductions, vectorized kernels, dtype conversions, and CPU scheduling |
| `simd.py` | Shared SIMD loop codegen for C++ and Triton-style iteration ranges, indexing, masks, and kernel features |
| `wrapper.py` | Python wrapper codegen, buffer allocation/reuse, kernel definition lines, kernel call lines, symbolic argument handling, and memory planning |
| `cpp_wrapper_cpu.py` | C++ wrapper generation for CPU and AOTI, including tensor handles, constants, and CPU Triton call wrappers |
| `cpp_wrapper_gpu.py` | C++ wrapper generation for CUDA/HIP/XPU style GPU kernels, cubin handling, streams, and AOTI integration |
| `multi_kernel.py` | Generates runtime selectable multi-kernel call sites and merges argument/workspace metadata across alternatives |
| `cpp_utils.py` | C++ expression printers, dtype maps, promotion helpers, local buffer context, and index-expression conversion |
| `triton_utils.py` | Triton signature, constexpr, tile-hint, and metadata utilities shared by Triton codegen paths |

## Public Interface

The public surface is internal to Inductor: `common.get_scheduling_for_device()` and backend registration select scheduling classes, `Kernel` subclasses emit loop code, and wrapper classes build executable module source. `TritonScheduling` and `CppScheduling` receive scheduler nodes and produce backend kernels. `PythonWrapperCodegen` emits Python callables for JIT mode, while `CppWrapperCpu` and GPU wrapper classes emit native wrapper code for AOTI and C++ wrapper modes. `MultiKernelState` and `MultiKernel` support runtime or size-hint based selection among generated alternatives.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Graph lowering and scheduling call backend codegen to emit kernels and wrappers |
| [torch/fx](torch/fx/ADR.md) | depends-on | Generated wrappers preserve FX node provenance, graph metadata, and traceback/debug artifacts |
| [c10/core](c10/core/ADR.md) | depends-on | Wrapper and kernel signatures encode tensor dtype, device, layout, symbolic size, stride, and storage information |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Extern and fallback code paths call ATen operations and use ATen tensor types in C++ wrappers |
| [torch/csrc/inductor](torch/csrc/inductor/ADR.md) | depended-on-by | AOTI runtime loads native wrappers, constants, cubins, and compiled kernels emitted here |
| Triton package | depends-on | `triton.py` emits `triton.jit` kernels and metadata consumed by Triton compilation |

## Runtime Behaviour

The scheduler hands fused `SchedulerNode` groups, extern-kernel nodes, and template choices to a backend scheduling class. Triton codegen converts loop bodies to `tl.load`, scalar math, reductions, masks, program ids, block shapes, and `tl.store`; C++ codegen converts the same logical loops to indexed C++ statements with vectorization and OpenMP reductions when legal. Wrapper codegen allocates outputs and workspaces, reuses compatible buffers, emits calls in dependency order, handles symbolic arguments, synchronizes device stream requirements, and returns user-visible tensors. Multi-kernel codegen emits a wrapper-level dispatcher that calls one of several compiled kernels based on autotune results or size-hint keys.

Generated source flows back to `torch/_inductor/codecache.py`, which compiles Python, C++, Triton, cubin, or AOTI artifacts and stores them under content-derived keys. At runtime the wrapper calls the compiled kernel functions directly, and extern-kernel wrapper lines call ATen or library operators for nodes that the fused backend does not own.

## Performance Profile

This directory determines much of the final kernel quality. `common.CSE` removes duplicate scalar expressions, `simd.py` and `cpp.py` select vectorized CPU expressions, `triton.py` chooses block sizes and reduction forms, and wrapper memory planning reduces allocation traffic by reusing buffers with matching device, dtype, size, alignment, and stream. Kernel launch count and memory bandwidth improve when scheduling fuses operations before codegen; register pressure, occupancy, vector width, and mask complexity then decide whether the generated kernel reaches hardware throughput. Compile-time cost comes from rendering large source strings, generating multiple templates, autotuning alternatives, and compiling through Triton or C++ toolchains. The wrapper layer adds minimal runtime overhead when cached because it executes straight-line allocation and kernel-call code with precomputed symbolic expressions.

## Design Rationale

Codegen is split by backend because Triton, CPU C++, AOTI wrappers, MPS, XPU, MTIA, Cutlass, and custom extern kernels have different syntax, launch semantics, dtype support, and optimization knobs. Shared abstractions in `common.py` and `simd.py` keep indexing, CSE, argument binding, workspace handling, and symbolic printing consistent across those backends. Wrapper generation is separate from kernel generation because allocation, mutation replay, stream choice, constants, and return-value packing are graph-level concerns rather than loop-body concerns. Source generation remains the interface to external compilers because it produces debuggable artifacts and lets Triton, clang/gcc, and vendor libraries perform final machine-specific optimization.
