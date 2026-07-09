# `functorch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch` is the public backward-compatibility package for PyTorch's functional transforms. Its implementation was merged into PyTorch core as `torch._functorch` and `torch.func`; this package re-exports the same APIs under the original `functorch.*` namespace and retains the `compile` and `dim` submodules.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `grad`, `vmap`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, `make_functional`, `functionalize`, `combine_state_for_ensemble` from `torch._functorch.deprecated`; exports `make_fx` from `torch._functorch.python_key` |
| `compile/__init__.py` | Re-exports `aot_function`, `aot_module`, `compiled_function`, `compiled_module`, `min_cut_rematerialization_partition` from `torch._functorch` |
| `_src/` | Legacy internal source retained for backward compatibility; mostly superseded by `torch/_functorch` |
| `dim/` | `Dim` — named-dimension tensor prototype (research-stage); not integrated into the main transform stack |
| `einops.py` | `rearrange`, `reduce`, `repeat` — einops-compatible operations using functorch's functional interface |
| `experimental/` | Experimental APIs: `chunk_vmap` (chunked vectorised map), `functionalize` |

## Public Interface

| Symbol | Description |
|---|---|
| `functorch.grad(fn)` | Deprecated alias for `torch.func.grad` |
| `functorch.vmap(fn)` | Deprecated alias for `torch.func.vmap` |
| `functorch.vjp(fn, *primals)` | Deprecated alias for `torch.func.vjp` |
| `functorch.make_functional(module)` | Extracts parameters from an `nn.Module` and returns a stateless function |
| `functorch.compile.aot_function(fn, fw_compiler, bw_compiler)` | Deprecated alias for `torch._functorch.aot_autograd.aot_function` |
| `functorch.compile.min_cut_rematerialization_partition` | Deprecated alias for `torch._functorch.partitioners.min_cut_rematerialization_partition` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | All implementations delegate to `torch._functorch`; `functorch` is a thin re-export layer |
| [torch/nn](torch/nn/ADR.md) | depends-on | `make_functional` extracts `nn.Module` parameters |
| `torch._functorch.deprecated` | depends-on | Deprecation-wrapped re-exports with `FutureWarning` on access |

## Runtime Behaviour

All calls to `functorch.*` APIs immediately delegate to the corresponding `torch._functorch.*` or `torch.func.*` implementation. `make_functional(module)` calls `torch._functorch.make_functional.make_functional`, which iterates `module.named_parameters()`, replaces each parameter with `None`, and returns a `FunctionalModule` that accepts the parameter list as a function argument. No transforms are applied here; it is a parameter-extraction helper. The `dim` submodule provides a prototype "named dimensions" API that uses a custom tensor subclass to track dimension identities; it is independent of the vmap transform.

## Performance Profile

- **Allocation sites**: `make_functional` performs one `None`-assignment per parameter; the returned `FunctionalModule` stores a reference to the original module. No tensor allocations occur at construction.
- **Synchronization costs**: none — all computation is delegated to `torch._functorch`.
- **Data movement**: no data movement in the re-export layer. `functorch.einops` performs reshape and permute operations which may or may not copy depending on stride layout.
- **Redundant or repeated work**: the deprecation warning machinery in `torch._functorch.deprecated` calls `warnings.warn` on first access per import, using a `set` of already-warned names to suppress repeats.

## Design Rationale

`functorch` exists as a standalone top-level package (rather than being removed) to avoid breaking downstream code that imports `functorch.grad`, `functorch.vmap`, etc. The implementation was moved to `torch/_functorch` to make it part of the core PyTorch install rather than a separate pip package, but the `functorch` namespace is retained for backward compatibility. New code should use `torch.func` instead.
