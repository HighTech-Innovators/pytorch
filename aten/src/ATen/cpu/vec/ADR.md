# `aten/src/ATen/cpu/vec`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/cpu/vec` owns ATen's CPU SIMD abstraction layer. It defines the `at::vec::Vectorized<T>` interface, architecture-specific vector headers, and the capability-dependent glue that higher-level CPU kernels use instead of writing raw intrinsics directly.

## Key Files

| File | Purpose |
|---|---|
| `vec_base.h` | Defines the generic `Vectorized<T>` container, width macros, type traits, and many fallback vector operations |
| `vec.h` | Selects the active architecture header set and provides shared helpers like `convert_to_bool(...)` and `Vectorized<bool>::loadu(...)` |
| `intrinsics.h` | Includes the header-only intrinsic bridge from `torch/headeronly/cpu/vec/intrinsics.h` |
| `vec256/vec256.h` | Provides 256-bit specializations used on non-AVX512 x86 builds |
| `vec512/vec512.h` | Provides 512-bit specializations used when `CPU_CAPABILITY_AVX512` is enabled |

## Public Interface

The main surface is `at::vec::Vectorized<T>`, including `size()`, `loadu(...)`, `blend<mask_>(...)`, `blendv(...)`, and `arange(...)`. `vec.h` also exposes `convert_to_bool(...)`, `VecHoldType`, and the `at::vec::CPU_CAPABILITY` inline namespace that selects the active instruction-set implementation.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/util](c10/util/ADR.md) | depends-on | `vec_base.h` uses `BFloat16`, `Half`, `Load.h`, `TypeCast.h`, and `c10::irange` utilities |
| [aten/src/ATen/native/cpu](aten/src/ATen/native/cpu/ADR.md) | depended-on-by | CPU kernels call this SIMD layer through helpers like `cpu_kernel_vec` and vectorized elementwise loops |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depended-on-by | Native operator implementations rely on the vector abstraction for backend-specific fast paths |

## Runtime Behaviour

`vec.h` selects `vec512/vec512.h` when `CPU_CAPABILITY_AVX512` is defined and otherwise includes `vec128/vec128.h` and `vec256/vec256.h`, so the compile target chooses the active instruction width before kernels instantiate templates. The same header defines `convert_to_bool(...)`, and `Vectorized<bool>::loadu(...)` uses an `int8_t` vector plus a stack buffer and `std::memcpy` to normalize boolean loads.

`vec_base.h` sets `VECTOR_WIDTH` to 64, 32, or 16 bytes depending on the active architecture macros, defines `Vectorized<T>::kSize` from that width, and provides generic operations such as `blendv(...)`, `arange(...)`, pointer conversion, and byte access. The file also declares `is_vec_specialized_for<T>` so generic kernels can detect whether a type has a hand-written SIMD specialization or should fall back to scalar-like logic.

## Performance Profile

- **Allocation sites** - Most operations stay on the stack; `Vectorized<T>` stores values in an aligned `std::array`, and helpers like `convert_to_bool(...)` and `blendv(...)` allocate temporary stack buffers for normalization and masking.
- **Synchronization costs** - The layer is lock-free and thread-agnostic; synchronization only happens in the higher-level parallel kernel loops that call these vector primitives.
- **Data movement** - `loadu(...)`, `store(...)`, `std::memcpy`, and aligned array copies dominate the abstraction, because the component exists to move scalar arrays into SIMD registers and back with the correct width and type conversions.
- **Redundant or repeated work** - `is_vec_specialized_for<T>` and the `CPU_CAPABILITY` inline namespace let kernels choose compile-time specializations instead of re-checking type support or ISA width inside every hot inner loop, and the note at the top of `vec_base.h` forbids static initializers so startup never pays AVX-enabled global-init cost.

## Design Rationale

ATen uses a templated SIMD abstraction because operator code needs one portable interface across AVX2, AVX512, SVE, and fallback paths. Centralizing vector semantics here lets `aten/src/ATen/native/cpu` implement kernels once against `Vectorized<T>` while architecture-specific headers hide the intrinsic details and width differences.
