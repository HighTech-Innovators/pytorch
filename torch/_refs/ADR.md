# `torch/_refs`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_refs` provides Python reference implementations and decompositions for existing PyTorch operators. These references make operator semantics explicit in Python so PrimTorch, decomposition passes, and tracing systems can lower complex APIs into simpler building blocks.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines the main catalog of Python reference operators and composes them with PrimTorch wrappers. |
| `_conversions.py` | Implements dtype conversion helpers plus decompositions for `aten.complex` and `aten.polar`. |
| `fft.py` | Implements FFT-family references by validating dtypes and shapes and lowering to primitive FFT operators. |
| `linalg/` | Holds additional reference linear algebra implementations used by PrimTorch remapping. |
| `nn/` | Holds reference implementations for neural network operators mapped from `torch.nn` and `torch.nn.functional`. |

## Public Interface

The directory exports hundreds of reference operators through `__all__`, including elementwise ops such as `add`, `mul`, and `where`, movement ops such as `clone` and `to`, reductions such as `sum` and `var`, and shape ops such as `view`, `reshape`, and `permute`. `_conversions.py` contributes `bfloat16`, `float`, `long`, `complex`, and `polar`, while `fft.py` exports `fft`, `ifft`, `rfft`, `irfft`, `fftn`, `ifftn`, `fftshift`, and `ifftshift`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_prims](torch/_prims/ADR.md) | mutual | Reference implementations call primitive ops such as `copy_to`, `resize`, `fft_r2c`, `fft_c2c`, and `fft_c2r`, while `torch._prims.context.TorchRefsMode` rewrites regular `torch.*` calls back into `_refs` during PrimTorch tracing. |
| [torch/_prims_common](torch/_prims_common/ADR.md) | depends-on | `__init__.py`, `_conversions.py`, and `fft.py` use shared type aliases, promotion helpers, and decorators such as `out_wrapper()` and `_maybe_convert_to_dtype()`. |

## Runtime Behaviour

The top-level reference functions in `__init__.py` are ordinary Python callables decorated with helpers like `elementwise_type_promotion_wrapper` and `out_wrapper`, so they validate and normalize arguments before composing simpler torch or prim operations. `_conversions.complex()` and `_conversions.polar()` are registered decompositions for `aten.complex` and `aten.polar`; they broadcast shapes, allocate result tensors with the correct complex dtype, and then fill `.real` and `.imag` fields explicitly.

`fft.py` validates normalization modes in `_apply_norm()`, promotes unsupported real and integral dtypes in `_promote_type_fft()`, and resizes or pads inputs in `_resize_fft_input()` before lowering to `prims.fft_r2c()`, `prims.fft_c2c()`, or `prims.fft_c2r()`. Those helpers are then assembled into public decompositions such as `fft()`, `ifft()`, `rfft()`, and `irfft()` with `@register_decomposition` and `@out_wrapper()`.

## Performance Profile

- **Allocation sites** - Reference implementations often allocate temporary tensors for promotion, padding, broadcasting, or `out=` compatibility because their job is to spell out semantics, not to be the minimal eager fast path.
- **Synchronization costs** - The references themselves do not synchronize devices, but they may trigger extra dispatcher traffic by composing several smaller ops where a native kernel would have executed one fused implementation.
- **Data movement** - Helpers like `_resize_fft_input()` can allocate padded copies, and conversion references such as `complex()` allocate a fresh result tensor before writing real and imaginary components into it.
- **Redundant or repeated work** - Python wrappers repeatedly perform promotion and metadata checks around every call, which is acceptable because these functions primarily serve tracing, decomposition, and correctness-oriented execution paths.

## Design Rationale

Reference implementations make semantics observable and composable at the Python level, which is essential for decomposition-driven compiler stacks. Keeping them separate from native kernels also lets PyTorch evolve the compiler lowering story without changing the public `torch.*` API surface or the underlying device kernels.
