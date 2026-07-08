# `torch/func`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/func` owns the public functional-transform API. It re-exports functorch-style transforms for vectorization, differentiation, functional module calls, and stateless random-number generation.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports functional transforms such as `grad`, `vmap`, `functionalize`, `jvp`, and `functional_call` |
| `_random.py` | Implements stateless Philox-based PRNG helpers such as `key`, `split`, `fold_in`, `normal`, and `uniform` |

## Public Interface

`grad`, `grad_and_value`, `vmap`, `replace_all_batch_norm_modules_`, `functionalize`, `hessian`, `jacfwd`, `jacrev`, `jvp`, `linearize`, `rearrange`, `vjp`, `functional_call`, `stack_module_state`, and `debug_unwrap` form the top-level API. `torch.func._random` adds `key`, `split`, `fold_in`, `normal_`, `normal`, `uniform_`, and `uniform` for explicit-key random generation.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | `__init__.py` directly re-exports transforms from `torch._functorch.apis`, `eager_transforms`, `functional_call`, and `einops` |
| [torch/nn](torch/nn/ADR.md) | depends-on | `functional_call` and `stack_module_state` operate on `nn.Module` parameters and buffers |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | `_random.py` calls `torch.ops.aten._philox_key_split`, `_philox_key_fold_in`, `_philox_normal_`, and `_philox_uniform_` |

## Runtime Behaviour

Importing `torch.func` fills the namespace by re-exporting transform implementations from the internal `_functorch` packages rather than defining new wrapper logic in place. `_random.key()` constructs a uint64 tensor `[seed, 0]`, which encodes the Philox seed and offset pair that the other stateless RNG helpers consume. `_random.split()` and `_random.fold_in()` call dedicated ATen operators to derive deterministic child keys without touching global generator state. `normal()` and `uniform()` allocate outputs on `key.device`, then forward to `normal_()` or `uniform_()` so the in-place ATen kernels do the actual sampling.

## Performance Profile

The top-level transform exports are effectively zero-cost aliases, so hot-path overhead comes from the underlying functorch transforms instead of this package. Stateless RNG avoids global generator mutation and associated synchronization, which makes it easier to batch and replay pure functions. The in-place variants `normal_()` and `uniform_()` let callers reuse existing storage, while `normal()` and `uniform()` always allocate a fresh tensor before sampling. Batched-key support pushes work into the ATen kernels, which avoids Python loops when many independent keys are needed.

## Design Rationale

The package makes functional transforms a first-class public API without exposing internal `_functorch` package names. Keeping stateless RNG beside differentiation and vectorization is intentional: all of these tools support writing pure, transformable functions that do not depend on hidden module or global state.
