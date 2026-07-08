# `c10/metal`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/metal` provides the shared Metal-side utility layer that ATen MPS host code and `.metal` kernels include directly. It owns common scalar metadata, indexing helpers, reductions, atomics, special math, RNG, and shader-side error reporting.

## Key Files

| File | Purpose |
|---|---|
| `common.h` | Defines shared constants such as `max_ndim`, `simdgroup_size`, `ILP_PER_THREAD`, and the `ScalarType` enum used by host and shader code |
| `indexing.h` | Implements coordinate-to-offset helpers plus reusable unary kernel templates such as `unary_dense`, `unary_strided`, and `unary_inner_contiguous` |
| `reduction_utils.h` | Implements `simd_sum`, `simd_prod`, `simd_argmin`, `simd_argmax`, and threadgroup reduction helpers |
| `atomic.h` | Specializes `AtomicType<T>` for float, integer, half, bfloat16, and bool atomics on Metal |
| `random.h` | Implements Philox-based `philox4::rand`, `rand`, `randn`, `box_muller_from_philox`, and `randint64` |
| `special_math.h` | Provides Metal implementations of `erf`, `erfc`, `erfinv`, `i0`, `i0e`, `i1`, and other math helpers |
| `error.h` | Defines `ErrorMessage`, `ErrorMessages`, `report_error`, and `TORCH_REPORT_ERROR` for shader error capture |

## Public Interface

Other components use `c10::metal::ScalarType`, `max_ndim`, `simdgroup_size`, `ILP_PER_THREAD`, `ceil_div`, and `round_up` from `common.h`. Kernel code calls `offset_from_coord()`, `pos_from_thread_index()`, `offset_from_thread_index()`, `unary_dense()`, `unary_strided()`, `threadgroup_sum()`, `simd_argmin()`, `simd_argmax()`, `AtomicType<T>::atomic_add()`, `philox4::rand()`, `randn()`, and `TORCH_REPORT_ERROR()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `aten/src/ATen/mps` (no ADR) | depended-on-by | Host-side MPS code includes `c10/metal/common.h` and `c10/metal/error.h` from files such as `EmptyTensor.cpp`, `MPSStream.mm`, and `OperationUtils.mm` |
| `aten/src/ATen/native/mps` (no ADR) | depended-on-by | MPS kernels include `indexing.h`, `atomic.h`, `reduction_utils.h`, `random.h`, `special_math.h`, and `utils.h` from files such as `UnaryKernel.metal`, `Distributions.metal`, and `ReduceOps.metal` |

## Runtime Behaviour

`common.h` fixes the shader ABI around `max_ndim = 16`, `simdgroup_size = 32`, and `ILP_PER_THREAD = 4`, then `indexing.h` uses those constants in `pos_from_thread_index()`, `offset_from_coord()`, and the `unary_dense()` and `unary_inner_contiguous()` kernels. `reduction_utils.h` performs warp-style reductions with `simd_shuffle_and_fill_down`, broadcasts winning lanes in `simd_argmin()` and `simd_argmax()`, and escalates to `threadgroup_barrier()` in `threadgroup_sum()` when more than one SIMD group participates. `random.h` generates Philox counters in `philox4::multiple_rounds()` and turns them into uniforms or Gaussian samples through `uint32_to_uniform_float()` and `box_muller_from_philox()`. `error.h` writes bounded error records into `ErrorMessages.msg` and advances `ErrorMessages.count` with an atomic fetch-add so shaders can report multiple failures safely.

## Performance Profile

- **Allocation sites** - The layer is header-only and mostly uses stack or threadgroup storage such as `array<T, ILP_PER_THREAD>` in `unary_dense()` and scratch buffers in `threadgroup_sum()`. `ErrorMessages` uses a fixed `msg[error_message_count]` array instead of per-error allocation.
- **Synchronization costs** - `threadgroup_sum()` and `threadgroup_sum2()` call `threadgroup_barrier()` when they combine partial results across SIMD groups. `AtomicType<half>`, `AtomicType<bfloat>`, `AtomicType<bool>`, and the small integer specializations spin in compare-exchange loops because Metal does not expose native atomics for those element types.
- **Data movement** - `unary_dense()` loads and stores four elements per thread through `ILP_PER_THREAD`, while `unary_inner_contiguous()` computes outer offsets once and then walks the contiguous inner span directly. `simd_broadcast()` and `simd_shuffle_and_fill_down()` keep reduction traffic inside a SIMD group before threadgroup memory is touched.
- **Redundant or repeated work** - The 2D `pos_from_thread_index(uint2, ...)` path removes one div/mod pair from every element in strided kernels. `simd_type<bfloat>` upcasts bfloat inputs to `float` and the 64-bit `simd_sum(long)` and `simd_prod(long)` fall back to explicit shuffle loops, so those paths do extra per-lane work to preserve correctness on Metal.

## Design Rationale

PyTorch keeps this code in `c10/metal` so every MPS kernel can share one set of type traits, math routines, and indexing templates instead of reimplementing them per operator. The files are header-only because ATen MPS kernels instantiate them per dtype and per operation from `.metal` sources, and the host side needs the same constants and enums when it names or configures those kernels.
