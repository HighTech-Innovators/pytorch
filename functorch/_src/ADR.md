# `functorch/_src`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch/_src` is a legacy private compatibility shim for functorch internals, mapped to book chapters 07 and 13 through its forwarding to `torch/_functorch`. The directory no longer owns the real transform implementations; each subpackage `__init__.py` states that the file moved under `torch/_functorch` and re-exports a small set of private names. It keeps older imports such as `functorch._src.vmap._add_batch_dim` or `functorch._src.aot_autograd.PytreeThunk` working for downstream code while centralizing maintenance in `torch/_functorch`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Empty package marker for the legacy private namespace |
| `aot_autograd/__init__.py` | Re-exports `aot_autograd_decompositions`, `KNOWN_TYPES`, and `PytreeThunk` from `torch._functorch.aot_autograd` |
| `eager_transforms/__init__.py` | Re-exports functional tensor assertion and unwrap helpers from `torch._functorch.eager_transforms` |
| `make_functional/__init__.py` | Re-exports `_swap_state` from `torch._functorch.make_functional` |
| `vmap/__init__.py` | Re-exports batching helpers, pytree helpers, `Tensor`, and validation utilities from `torch._functorch.vmap` |

## Public Interface

This namespace is private and compatibility-only. The exported names are the exact imports listed in each subpackage file: AOTAutograd decomposition/type helpers, eager transform functional-tensor helpers, `_swap_state`, and vmap internals such as `_add_batch_dim`, `_remove_batch_dim`, `_unwrap_batched`, `_process_batched_inputs`, `_validate_and_get_batch_size`, `tree_flatten`, and `tree_unflatten`. The files warn in comments that non-PyTorch developers relying on these imports should file an issue.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Every non-empty file imports implementation symbols from `torch._functorch` |
| [functorch/compile](functorch/compile/ADR.md) | related | Both directories preserve legacy functorch import paths after implementation moved to `torch._functorch` |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | Re-exported AOTAutograd and vmap helpers operate on FX and pytree structures through their real implementations |

## Runtime Behaviour

Importing a `functorch._src` submodule executes only the corresponding `__init__.py` forwarding imports. No tracing, batching, grad, functionalization, partitioning, or graph compilation logic lives in this directory. When downstream code calls a re-exported helper, execution immediately enters the implementation in `torch._functorch`, such as vmap's BatchedTensor add/remove helpers or AOTAutograd's pytree thunk utilities. The empty top-level `__init__.py` exists only to make the namespace importable.

Because the subpackages expose private helpers, import compatibility matters for tests and downstream code that reached into old functorch internals. The shim does not validate arguments or adapt behavior; it preserves identity with the moved symbols as much as normal Python imports allow.

## Performance Profile

The directory adds negligible overhead: each file performs a handful of imports at module load time and then leaves runtime work to `torch/_functorch`. There are no hot loops, no caches, no graph data structures, and no dispatch rules here. Performance-sensitive behavior of re-exported functions remains in the real implementation, where vmap manipulates dynamic layers and AOTAutograd captures and partitions FX graphs. Keeping this shim minimal prevents legacy imports from adding extra call frames to transform execution.

## Design Rationale

`functorch/_src` remains as a compatibility layer because private imports existed before functorch functionality moved into PyTorch proper. Deleting the namespace would break downstream users and internal tests even though the real code now lives in `torch/_functorch`. Forwarding through `__init__.py` files keeps migration risk low and makes ownership clear: new behavior belongs in `torch/_functorch`, not in the legacy package. The sparse directory also avoids maintaining duplicate private APIs across two implementation trees.
