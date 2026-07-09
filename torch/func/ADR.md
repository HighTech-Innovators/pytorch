# `torch/func`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/func` exposes PyTorch's function-transform API as a public namespace over `torch._functorch` and related functorch functionality. The package re-exports transforms such as `grad`, `vmap`, `jacrev`, `jacfwd`, `vjp`, `jvp`, `functionalize`, and module-state helpers, and it also contains experimental stateless PRNG helpers in `_random.py`.

## Key Files

| File | Purpose |
|---|---|
| `torch/func/__init__.py` | Public shim that imports function transforms from `torch._functorch` modules and lists them in `__all__`. |
| `torch/func/_random.py` | Experimental stateless PRNG API with `key`, `split`, `fold_in`, `normal_`, `normal`, `uniform_`, and `uniform`. |

## Public Interface

| Symbol | Description |
|---|---|
| `grad`, `grad_and_value`, `vmap` | Imported from `torch._functorch.apis` for automatic differentiation and batching transforms. |
| `functionalize`, `hessian`, `jacfwd`, `jacrev`, `jvp`, `linearize`, `vjp`, `debug_unwrap` | Imported from `torch._functorch.eager_transforms`. |
| `functional_call`, `stack_module_state` | Imported from `torch._functorch.functional_call` for calling modules with explicit state and stacking module parameters or buffers. |
| `replace_all_batch_norm_modules_` | Imported from `torch._functorch.batch_norm_replacement` for transform-friendly BatchNorm mutation. |
| `rearrange` | Imported from `torch._functorch.einops`. |
| `torch.func._random.key(seed, impl="philox4x32-10", device=None)` | Creates a uint64 `(seed, offset)` Philox key tensor. |
| `torch.func._random.split(key, num=2)` and `fold_in(key, data)` | Derive stateless PRNG keys through `torch.ops.aten._philox_key_split` and `_philox_key_fold_in`. |
| `torch.func._random.normal`, `normal_`, `uniform`, `uniform_` | Allocate or fill tensors using Philox ATen operators and the supplied key. |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | `__init__.py` imports every public transform from `torch._functorch.apis`, `eager_transforms`, `functional_call`, `einops`, and `batch_norm_replacement`. |
| [functorch](functorch/ADR.md) | depends-on | Provides the historical transform implementation and compatibility context behind the `torch.func` API surface. |
| [torch](torch/ADR.md) | depends-on | `_random.py` uses `torch.tensor`, `torch.empty`, dtypes, devices, and ATen ops under `torch.ops.aten`. |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Stateless PRNG helpers call `_philox_key_split`, `_philox_key_fold_in`, `_philox_normal_`, and `_philox_uniform_` ATen operators. |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Transform APIs such as `grad`, `grad_and_value`, `jacrev`, `hessian`, `vjp`, and `jvp` build on PyTorch autograd semantics. |

## Runtime Behaviour

Importing `torch.func` binds names directly from `torch._functorch` modules; the shim does not wrap `grad`, `vmap`, `functionalize`, `functional_call`, or the Jacobian and Hessian helpers locally. `_random.key` validates that `impl` equals `"philox4x32-10"` and creates a `torch.uint64` tensor containing `[seed, 0]` on the requested device. `_random.split` and `_random.fold_in` delegate key derivation to ATen Philox operators, while `normal` and `uniform` normalize a shape argument, allocate an output tensor on `key.device`, default `dtype` to `torch.float32`, and call the in-place `normal_` or `uniform_` helper.

## Performance Profile

The public transform names incur only import-time aliasing cost in `torch/func/__init__.py`; execution cost belongs to the `torch._functorch` implementations that perform batching, functionalization, and autodiff transforms. `torch.func._random.normal` and `uniform` allocate a fresh output tensor before calling the in-place ATen Philox kernels, while `normal_` and `uniform_` avoid that allocation by filling an existing result tensor. `split` and `fold_in` support batched keys and run through ATen operators, so they avoid Python loops over keys. The PRNG key itself is a small `torch.uint64` tensor of shape `(2,)`, which keeps state explicit and device-local.

## Design Rationale

The package makes the function-transform namespace stable without duplicating the implementation that already lives in `torch._functorch`. The `__all__` list documents the supported transform surface and keeps compatibility with the functorch migration path. The `_random.py` module demonstrates explicit, stateless PRNG keys so transforms can generate reproducible random values without mutating global generator state.
