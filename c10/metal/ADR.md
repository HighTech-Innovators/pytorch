# `c10/metal`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`c10/metal` provides shared Metal shader utilities for PyTorch's MPS and sparse MPS kernels. It defines type traits, scalar type constants, indexing helpers, reduction primitives, atomics, random helpers, and special math functions that Metal kernels include directly.

## Key Files

| File | Purpose |
|---|---|
| `common.h` | Defines `max_ndim`, `simdgroup_size`, `ILP_PER_THREAD`, `ScalarType`, `ceil_div()`, and `round_up()` for host and Metal code |
| `utils.h` | Defines vector type traits, `OpMathType`, `AccumulationType`, `min`, `max`, `less`, `vec2type_t`, `vec4type_t`, and `opmath_t` |
| `indexing.h` | Defines coordinate-to-offset helpers and generic unary Metal kernels such as `unary_dense`, `unary_strided`, and `unary_inner_contiguous` |
| `special_math.h` | Implements Metal versions of `erf`, `erfc`, `erfinv`, `i0`, `i0e`, `i1`, `i1e`, and `log_gamma` |
| `igamma.h` | Ports regularized incomplete gamma helper routines, including `ratevl()` and Lanczos-based helpers |
| `reduction_utils.h` | Implements SIMD and threadgroup reductions such as `simd_sum`, `simd_prod`, `simd_min`, `simd_max`, `threadgroup_sum`, and argmin/argmax helpers |

## Public Interface

| Symbol | Description |
|---|---|
| `c10::metal::ScalarType` | Metal-side scalar type enum generated from `C10_METAL_ALL_TYPES_FUNCTOR` |
| `c10::metal::max_ndim`, `simdgroup_size`, `ILP_PER_THREAD` | Compile-time constants used by indexing and reduction kernels |
| `offset_from_coord()` / `pos_from_thread_index()` | Convert tensor coordinates and thread positions into strided offsets |
| `unary_dense`, `unary_strided`, `unary_inner_contiguous` | Template Metal kernels for dense, general strided, and inner-contiguous unary elementwise execution |
| `REGISTER_UNARY_OP` | Macro that instantiates dense, scalar, strided, inner-contiguous, and castout unary kernel variants with `host_name` attributes |
| `opmath_t`, `accum_t`, `vec2type_t`, `vec4type_t` | Type aliases used to promote low-precision inputs and select Metal vector types |
| `simd_sum`, `simd_prod`, `simd_min`, `simd_max`, `threadgroup_sum` | Reduction helpers built on Metal SIMD-group operations |
| `erf`, `erfc`, `erfinv`, `i0`, `i0e`, `i1`, `i1e`, `log_gamma` | Special math functions used by Metal activation, binary, comparison, and distribution kernels |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen MPS and sparse MPS kernels include `c10/metal/indexing.h`, `utils.h`, `special_math.h`, `reduction_utils.h`, and `atomic.h` |
| [c10/core](c10/core/ADR.md) | depended-on-by | c10 device and scalar abstractions expose MPS tensors whose kernels use these Metal utilities |
| [torch](torch/ADR.md) | depended-on-by | Python MPS tensor operations ultimately execute ATen Metal kernels that include these headers |

## Runtime Behaviour

Metal kernel sources include these headers at shader compile time, so most functions run in device code as inlined templates. `indexing.h` computes offsets from runtime `sizes`, `input_strides`, and `output_strides` buffers, then dispatches functors through dense, strided, or inner-contiguous kernels. `reduction_utils.h` uses `simd_shuffle_and_fill_down`, `simd_ballot`, and `threadgroup_barrier` to combine per-lane values across SIMD groups and threadgroups.

## Performance Profile

`ILP_PER_THREAD` is set to 4 in `common.h`, and `unary_dense` loads four elements per thread into thread-local arrays to improve memory-level parallelism for contiguous tensors. `unary_strided` uses a 2D grid so `thread_pos.x` maps to the fastest-varying dimension and saves one division and modulus per element compared with a purely linear decomposition. `reduction_utils.h` promotes `half` and `bfloat` to `float` through `OpMathType` and `AccumulationType`, which improves numerical behavior while increasing per-element compute width. Long integer reductions use `int2` shuffles because Metal does not provide native SIMD reductions for 64-bit types.

## Design Rationale

The directory keeps Metal shader support in c10 so ATen kernels can share small, header-only utilities without depending on higher-level Python or dispatcher code. Generic templates and explicit instantiation macros let many operations reuse the same indexing, casting, and reduction kernels while preserving Metal `host_name` entry points. The special math implementations mirror CUDA and CPU formulas where possible, giving MPS kernels compatible numerical behavior without calling unavailable device-library functions.
